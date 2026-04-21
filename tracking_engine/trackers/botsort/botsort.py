# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from pathlib import Path
from typing import Any

import numpy as np
import torch

from tracking_engine.motion.cmc import get_cmc_method
from tracking_engine.motion.kalman_filters.xywh import KalmanFilterXYWH
from tracking_engine.reid.core import ReID
from tracking_engine.trackers.basetracker import BaseTracker
from tracking_engine.trackers.botsort.basetrack import BaseTrack, TrackState
from tracking_engine.trackers.botsort.botsort_track import STrack
from tracking_engine.trackers.botsort.botsort_utils import (joint_stracks,
                                                   remove_duplicate_stracks,
                                                   sub_stracks)
from tracking_engine.utils import logger as LOGGER
from tracking_engine.utils.matching import (embedding_distance, fuse_score,
                                   iou_distance, linear_assignment)


class BotSort(BaseTracker):
    """Initialize the BotSort tracker.

    Args:
        reid_weights (Path): Path to the ReID model weights.
        device (torch.device): Device used for ReID inference.
        half (bool): Whether to use half precision for ReID inference.
        track_high_thresh (float): Confidence threshold for the first
            association pass.
        track_low_thresh (float): Lower confidence bound for candidate
            detections.
        new_track_thresh (float): Threshold required to initialize a new track.
        track_buffer (int): Number of frames to keep unmatched tracks alive.
        match_thresh (float): Matching threshold used during association.
        proximity_thresh (float): IoU gate used before appearance matching.
        appearance_thresh (float): Maximum embedding distance accepted for ReID
            matching.
        reid_recovery_enabled (bool): Whether to run an additional ReID-only
            recovery pass for unmatched lost tracks before creating new IDs.
        reid_recovery_proximity_thresh (float): IoU-distance gate used in the
            ReID recovery pass before combining with embedding distance.
        reid_recovery_appearance_thresh (float): Embedding-distance gate used
            specifically in the ReID recovery pass.
        reid_recovery_thresh (float): Embedding-distance threshold used in the
            ReID recovery pass.
        second_association_thresh (float): Matching threshold for the low-score
            second association stage.
        unconfirmed_association_thresh (float): Matching threshold for
            unconfirmed tracks association.
        cmc_method (str): Camera-motion compensation method.
        frame_rate (int): Frame rate used to scale the internal track buffer.
        fuse_first_associate (bool): Whether to fuse motion and appearance in
            the first association step.
        with_reid (bool): Whether to enable appearance features.
        **kwargs: Base tracker settings forwarded to :class:`BaseTracker`,
            including ``det_thresh``, ``max_age``, ``max_obs``, ``min_hits``,
            ``iou_threshold``, ``per_class``, ``nr_classes``, ``asso_func``,
            and ``is_obb``.

    Attributes:
        lost_stracks (list[STrack]): Tracks kept in the lost state.
        removed_stracks (list[STrack]): Tracks removed from the tracker state.
        buffer_size (int): Track buffer size after frame-rate scaling.
        max_time_lost (int): Maximum number of frames a track may stay lost.
        kalman_filter (KalmanFilterXYWH): Motion model used for prediction.
        model: ReID model used for appearance extraction when enabled.
        cmc: Camera-motion compensation method.
    """

    supports_obb = True

    def __init__(
        self,
        reid_weights: Path,
        device: torch.device,
        half: bool,
        # BotSort-specific parameters
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        reid_recovery_enabled: bool = True,
        reid_recovery_proximity_thresh: float = 0.5,
        reid_recovery_appearance_thresh: float = 0.25,
        reid_recovery_thresh: float = 0.35,
        second_association_thresh: float = 0.5,
        unconfirmed_association_thresh: float = 0.7,
        cmc_method: str = "ecc",
        frame_rate: int = 30,
        fuse_first_associate: bool = False,
        with_reid: bool = True,
        **kwargs: Any,  # BaseTracker parameters
    ):
        # Capture all init params for logging
        init_args = {k: v for k, v in locals().items() if k not in ('self', 'kwargs')}
        super().__init__(**init_args, _tracker_name='BotSort', **kwargs)
        
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.reid_recovery_enabled = bool(reid_recovery_enabled)
        self.reid_recovery_proximity_thresh = float(
            max(reid_recovery_proximity_thresh, 0.0)
        )
        self.reid_recovery_appearance_thresh = float(
            max(reid_recovery_appearance_thresh, 0.0)
        )
        self.reid_recovery_thresh = float(max(reid_recovery_thresh, 0.0))
        self.second_association_thresh = second_association_thresh
        self.unconfirmed_association_thresh = unconfirmed_association_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilterXYWH(ndim=5 if self.is_obb else 4)

        # ReID module
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.with_reid = with_reid
        if self.with_reid:
            self.model = ReID(
                weights=reid_weights, device=device, half=half
            ).model

        cmc_cls = get_cmc_method(cmc_method)
        self.cmc = cmc_cls() if cmc_cls is not None else None
        self.fuse_first_associate = fuse_first_associate

    def _kalman_ndim(self) -> int:
        return self.detection_layout.box_cols

    def _detection_boxes(self, dets: np.ndarray) -> np.ndarray:
        return self.detection_layout.boxes(dets)

    def _track_box_centers(self, tracks: list[STrack]) -> np.ndarray:
        if len(tracks) == 0:
            return np.empty((0, 2), dtype=np.float32)

        if self.is_obb:
            centers = np.asarray([t.xywha[:2] for t in tracks], dtype=np.float32)
        else:
            boxes = np.asarray([t.xyxy for t in tracks], dtype=np.float32)
            centers = np.column_stack(
                ((boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5)
            ).astype(np.float32)
        return centers

    def _normalized_center_l2_distance(
        self,
        tracks: list[STrack],
        detections: list[STrack],
    ) -> np.ndarray:
        """Khoảng cách L2 giữa tâm bbox, chuẩn hóa theo đường chéo frame về [0, 1]."""
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((len(tracks), len(detections)), dtype=np.float32)

        track_centers = self._track_box_centers(tracks)
        det_centers = self._track_box_centers(detections)

        dists = np.linalg.norm(
            track_centers[:, None, :] - det_centers[None, :, :], axis=2
        )
        frame_diag = float(np.hypot(getattr(self, "w", 1.0), getattr(self, "h", 1.0)))
        if frame_diag <= 0:
            frame_diag = 1.0

        return np.clip(dists / frame_diag, 0.0, 1.0).astype(np.float32)

    def _obb_detections_to_cmc_boxes(self, dets: np.ndarray) -> np.ndarray:
        """Convert OBB detections to enclosing AABBs for CMC feature masking."""
        if len(dets) == 0:
            return np.empty((0, 4), dtype=np.float32)
        return np.asarray(
            [STrack.obb_to_xyxy(det[:5]) for det in dets], dtype=np.float32
        )

    def _apply_aabb_camera_motion_compensation(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        strack_pool: list[STrack],
        unconfirmed: list[STrack],
    ) -> None:
        """Apply the legacy BoTSORT CMC path for axis-aligned tracks."""
        warp = self.cmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

    def _apply_obb_camera_motion_compensation(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        strack_pool: list[STrack],
        unconfirmed: list[STrack],
    ) -> None:
        """Apply OBB-specific CMC using enclosing AABBs for estimation."""
        warp = self.cmc.apply(img, self._obb_detections_to_cmc_boxes(dets))
        STrack.multi_gmc_obb(strack_pool, warp)
        STrack.multi_gmc_obb(unconfirmed, warp)

    def _apply_camera_motion_compensation(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        strack_pool: list[STrack],
        unconfirmed: list[STrack],
    ) -> None:
        """Dispatch camera motion compensation without mixing AABB and OBB logic."""
        if self.cmc is None:
            return
        if self.is_obb:
            self._apply_obb_camera_motion_compensation(
                dets, img, strack_pool, unconfirmed
            )
            return
        self._apply_aabb_camera_motion_compensation(
            dets, img, strack_pool, unconfirmed
        )

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(
        self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None
    ) -> np.ndarray:
        self.check_inputs(dets, img, embs)
        self.kalman_filter = KalmanFilterXYWH(ndim=self._kalman_ndim())
        self.frame_count += 1
        self._new_track_debug: dict[int, dict[str, str]] = {}

        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        # Preprocess detections
        dets, dets_first, embs_first, dets_second = self._split_detections(dets, embs)

        # Extract appearance features
        if self.with_reid and embs is None:
            features_high = self.model.get_features(self._detection_boxes(dets_first), img)
        else:
            features_high = embs_first if embs_first is not None else []

        # Create detections
        detections = self._create_detections(dets_first, features_high)

        # Separate unconfirmed and active tracks
        unconfirmed, active_tracks = self._separate_tracks()

        strack_pool = joint_stracks(active_tracks, self.lost_stracks)

        # First association
        matches_first, u_track_first, u_detection_first = self._first_association(
            dets,
            dets_first,
            active_tracks,
            unconfirmed,
            img,
            detections,
            activated_stracks,
            refind_stracks,
            strack_pool,
        )

        # Second association
        matches_second, u_track_second, u_detection_second = self._second_association(
            dets_second,
            activated_stracks,
            lost_stracks,
            refind_stracks,
            u_track_first,
            strack_pool,
        )

        # Handle unconfirmed tracks
        matches_unc, u_track_unc, u_detection_unc = self._handle_unconfirmed_tracks(
            u_detection_first,
            detections,
            activated_stracks,
            removed_stracks,
            unconfirmed,
        )

        # Extra ReID recovery for lost tracks before creating brand-new IDs
        u_detection_unc = self._reid_recovery_association(
            u_track_first,
            u_detection_first,
            u_detection_unc,
            detections,
            refind_stracks,
            strack_pool,
        )

        # Initialize new tracks
        self._initialize_new_tracks(
            u_detection_unc,
            activated_stracks,
            [detections[i] for i in u_detection_first],
        )

        # Update lost and removed tracks
        self._update_track_states(removed_stracks)

        # Merge and prepare output
        return self._prepare_output(
            activated_stracks, refind_stracks, lost_stracks, removed_stracks
        )

    def _split_detections(self, dets, embs):
        dets = self.detection_layout.with_detection_indices(dets)
        confs = self.detection_layout.confidences(dets)
        second_mask = np.logical_and(
            confs > self.track_low_thresh, confs < self.track_high_thresh
        )
        dets_second = dets[second_mask]
        first_mask = confs > self.track_high_thresh
        dets_first = dets[first_mask]
        embs_first = embs[first_mask] if embs is not None else None
        return dets, dets_first, embs_first, dets_second

    def _create_detections(self, dets_first, features_high):
        if len(dets_first) > 0:
            if self.with_reid:
                detections = [
                    STrack(det, f, max_obs=self.max_obs, is_obb=self.is_obb)
                    for (det, f) in zip(dets_first, features_high)
                ]
            else:
                detections = [STrack(det, max_obs=self.max_obs, is_obb=self.is_obb) for det in dets_first]
        else:
            detections = []
        return detections

    def _separate_tracks(self):
        unconfirmed, active_tracks = [], []
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                active_tracks.append(track)
        return unconfirmed, active_tracks

    def _first_association(
        self,
        dets,
        dets_first,
        active_tracks,
        unconfirmed,
        img,
        detections,
        activated_stracks,
        refind_stracks,
        strack_pool,
    ):

        STrack.multi_predict(strack_pool)

        # Fix camera motion
        self._apply_camera_motion_compensation(
            dets, img, strack_pool, unconfirmed
        )

        # Associate with high confidence detection boxes
        ious_dists = iou_distance(strack_pool, detections, is_obb=self.is_obb)
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_first_associate:
            ious_dists = fuse_score(ious_dists, detections)

        raw_emb_dists = None
        if self.with_reid:
            emb_dists = embedding_distance(strack_pool, detections)
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        for idet in u_detection:
            det = detections[idet]
            det_key = int(det.det_ind)
            if dists.shape[0] == 0:
                first_msg = "first_association unmatched: no tracks in strack_pool"
            else:
                best_cost = float(np.min(dists[:, idet]))
                first_msg = (
                    "first_association unmatched: best_cost={:.3f} > "
                    "match_thresh({:.3f})"
                ).format(best_cost, float(self.match_thresh))
                if self.with_reid and raw_emb_dists is not None:
                    proximity_blocked = bool(
                        np.all(ious_dists[:, idet] > self.proximity_thresh)
                    )
                    appearance_blocked = bool(
                        np.all(raw_emb_dists[:, idet] > self.appearance_thresh)
                    )
                    first_msg += (
                        "; proximity_blocked={}; appearance_blocked={}"
                    ).format(proximity_blocked, appearance_blocked)
            self._new_track_debug.setdefault(det_key, {})["first"] = first_msg

        return matches, u_track, u_detection

    def _second_association(
        self,
        dets_second,
        activated_stracks,
        lost_stracks,
        refind_stracks,
        u_track_first,
        strack_pool,
    ):
        if len(dets_second) > 0:
            detections_second = [
                STrack(det, max_obs=self.max_obs, is_obb=self.is_obb) for det in dets_second
            ]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track_first
            if strack_pool[i].state == TrackState.Tracked
        ]

        dists = iou_distance(r_tracked_stracks, detections_second, is_obb=self.is_obb)
        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.second_association_thresh
        )

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        return matches, u_track, u_detection

    def _handle_unconfirmed_tracks(
        self, u_detection, detections, activated_stracks, removed_stracks, unconfirmed
    ):
        """
        Handle unconfirmed tracks (tracks with only one detection frame).

        Args:
            u_detection: Unconfirmed detection indices.
            detections: Current list of detections.
            activated_stracks: List of newly activated tracks.
            removed_stracks: List of tracks to remove.
        """
        # Only use detections that are unconfirmed (filtered by u_detection)
        detections = [detections[i] for i in u_detection]

        # Calculate IoU distance between unconfirmed tracks and detections
        ious_dists = iou_distance(unconfirmed, detections, is_obb=self.is_obb)

        ious_dists_mask = ious_dists > self.proximity_thresh
        ious_dists = fuse_score(ious_dists, detections)

        # Fuse scores for IoU-based and embedding-based matching (if applicable)
        if self.with_reid:
            emb_dists = embedding_distance(unconfirmed, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        # Perform data association using linear assignment on the combined distances
        matches, u_unconfirmed, u_detection = linear_assignment(
            dists, thresh=self.unconfirmed_association_thresh
        )

        # Update matched unconfirmed tracks
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_count)
            activated_stracks.append(unconfirmed[itracked])

        # Mark unmatched unconfirmed tracks as removed
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        for idet in u_detection:
            det = detections[idet]
            det_key = int(det.det_ind)
            if dists.shape[0] == 0:
                unc_msg = "unconfirmed_association unmatched: no unconfirmed tracks"
            else:
                best_cost = float(np.min(dists[:, idet]))
                unc_msg = (
                    "unconfirmed_association unmatched: best_cost={:.3f} > "
                    "unconfirmed_association_thresh({:.3f})"
                ).format(best_cost, float(self.unconfirmed_association_thresh))
            self._new_track_debug.setdefault(det_key, {})["unconfirmed"] = unc_msg

        return matches, u_unconfirmed, u_detection

    def _reid_recovery_association(
        self,
        u_track_first,
        u_detection_first,
        u_detection_unc,
        detections,
        refind_stracks,
        strack_pool,
    ):
        if not self.with_reid or not self.reid_recovery_enabled:
            return u_detection_unc
        if len(u_detection_unc) == 0:
            return u_detection_unc

        lost_candidates = [
            strack_pool[i]
            for i in u_track_first
            if strack_pool[i].state == TrackState.Lost
        ]
        if len(lost_candidates) == 0:
            for idx in u_detection_unc:
                det_key = int(detections[u_detection_first[idx]].det_ind)
                self._new_track_debug.setdefault(det_key, {})["reid_recovery"] = (
                    "reid_recovery unmatched: no lost-track candidates"
                )
            return u_detection_unc

        det_indices = [u_detection_first[i] for i in u_detection_unc]
        recovery_dets = [detections[i] for i in det_indices]

        ious_dists = self._normalized_center_l2_distance(lost_candidates, recovery_dets)
        # ious_dists_mask = ious_dists > self.reid_recovery_proximity_thresh

        emb_dists = embedding_distance(lost_candidates, recovery_dets)
        raw_emb = emb_dists.copy()
        emb_dists[emb_dists > self.reid_recovery_appearance_thresh] = 1.0
        # emb_dists[ious_dists_mask] = 1.0
        dists = 1.2 * ious_dists + 0.2 * emb_dists
        matches, u_lost, u_det = linear_assignment(
            dists, thresh=self.reid_recovery_thresh
        )

        for ilost, idet in matches:
            track = lost_candidates[ilost]
            det = recovery_dets[idet]
            track.re_activate(det, self.frame_count, new_id=False)
            refind_stracks.append(track)

        for local_idx, det in enumerate(recovery_dets):
            det_key = int(det.det_ind)
            if dists.shape[0] == 0:
                msg = "reid_recovery unmatched: no valid embedding rows"
            else:
                best_cost = float(np.min(dists[:, local_idx]))
                raw_best = float(np.min(raw_emb[:, local_idx]))
                msg = (
                    "reid_recovery unmatched: best_cost={:.3f} (raw={:.3f}) > "
                    "reid_recovery_thresh({:.3f})"
                ).format(best_cost, raw_best, float(self.reid_recovery_thresh))
                proximity_blocked = bool(
                    np.all(
                        ious_dists[:, local_idx]
                        > self.reid_recovery_proximity_thresh
                    )
                )
                appearance_blocked = bool(
                    np.all(
                        raw_emb[:, local_idx]
                        > self.reid_recovery_appearance_thresh
                    )
                )
                msg += "; proximity_blocked={}; appearance_blocked={}".format(
                    proximity_blocked, appearance_blocked
                )
            self._new_track_debug.setdefault(det_key, {})["reid_recovery"] = msg

        remaining_local = [int(i) for i in u_det]
        return [u_detection_unc[i] for i in remaining_local]

    def _initialize_new_tracks(self, u_detections, activated_stracks, detections):
        for inew in u_detections:
            track = detections[inew]
            if track.conf < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_count)
            activated_stracks.append(track)
            debug_info = self._new_track_debug.get(int(track.det_ind), {})
            first_reason = debug_info.get("first", "first_association reason unavailable")
            unc_reason = debug_info.get(
                "unconfirmed", "unconfirmed_association reason unavailable"
            )
            recovery_reason = debug_info.get(
                "reid_recovery", "reid_recovery reason unavailable"
            )
            LOGGER.info(
                "New track ID {} created at frame {} (det_ind={}, conf={:.3f}). "
                "Reason: {}; {}; {}; and conf >= new_track_thresh({:.3f}).",
                int(track.id),
                int(self.frame_count),
                int(track.det_ind),
                float(track.conf),
                first_reason,
                unc_reason,
                recovery_reason,
                float(self.new_track_thresh),
            )

    def _update_tracks(
        self,
        matches,
        strack_pool,
        detections,
        activated_stracks,
        refind_stracks,
        mark_removed=False,
    ):
        # Update or reactivate matched tracks
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        # Mark only unmatched tracks as removed, if mark_removed flag is True
        if mark_removed:
            unmatched_tracks = [
                strack_pool[i]
                for i in range(len(strack_pool))
                if i not in [m[0] for m in matches]
            ]
            for track in unmatched_tracks:
                track.mark_removed()

    def _update_track_states(self, removed_stracks):
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

    def _prepare_output(
        self, activated_stracks, refind_stracks, lost_stracks, removed_stracks
    ):
        self.active_tracks = [
            t for t in self.active_tracks if t.state == TrackState.Tracked
        ]
        self.active_tracks = joint_stracks(self.active_tracks, activated_stracks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(
            self.active_tracks, self.lost_stracks
        )

        outputs = [
            [*(t.xywha if self.is_obb else t.xyxy), t.id, t.conf, t.cls, t.det_ind]
            for t in self.active_tracks
            if t.is_activated
        ]

        return np.asarray(outputs, dtype=np.float32) if outputs else self.empty_output(dtype=np.float32)

