import copy
from dataclasses import replace

import numpy as np
import pytest
import stereo_training_factors as stf

from stereo_pose_arbitration import SupportedStereoFrame
from stereo_training_factors import (
    EndpointPose,
    accumulate_schur_information,
    build_stereo_training_factors,
    factor_schur_information,
    linearize_training_factor,
    motion_rotation_information,
    se3_exp,
    se3_adjoint,
    stereo_projection,
)


K = np.array([[510., 0.7, 320.], [0., 505., 240.], [0., 0., 1.]])
B = .54
OFFSET = .7
SIZE = (640, 480)
CALIB = "synthetic-calib-v1"
EPOCH = (12, 4)


def _pose(xi=None):
    return np.eye(4) if xi is None else se3_exp(np.asarray(xi, float))


def _project(T, X):
    p = T[:3, :3].T @ (X-T[:3, 3])
    return stereo_projection(p, K, B, OFFSET)[0]


def _scene(n=32, *, broad=True, lm_ids=None, duplicates=0, target_points_finite=False):
    if broad:
        xs = np.linspace(-3.8, 3.8, n)
        ys = 1.25*np.sin(np.arange(n)*.73)
    else:
        xs = np.linspace(-.18, .18, n)
        ys = .14*np.sin(np.arange(n)*.73)
    zs = 8. + (np.arange(n) % 7)*1.1
    world = np.c_[xs, ys, zs]
    anchor0 = _pose([.08, -.03, .04, .018, -.011, .023])
    source_rel = _pose([.13, -.015, .02, .04, .02, -.018])
    source_pose = anchor0 @ source_rel
    target_pose = _pose([.31, .025, .055, .049, -.012, .031])
    source_camera = (world-source_pose[:3, 3]) @ source_pose[:3, :3]
    target_camera = (world-target_pose[:3, 3]) @ target_pose[:3, :3]
    source_obs = np.asarray([_project(source_pose, X) for X in world])
    target_obs = np.asarray([_project(target_pose, X) for X in world])
    # Fixed, declared measurement noise, identical across narrow/broad scenes.
    noise = .025*np.c_[np.sin(np.arange(n)), np.cos(np.arange(n)*.5), np.sin(np.arange(n)*.3)]
    source_obs += noise
    target_obs -= noise*.5
    sid = np.full(n, -1, np.int32) if lm_ids is None else np.asarray(lm_ids, np.int32).copy()
    tid = np.full(n, -1, np.int32) if lm_ids is None else np.asarray(lm_ids, np.int32).copy()
    s_pixels, t_pixels = source_obs[:, :2], target_obs[:, :2]
    s_points, t_points = source_camera, target_camera if target_points_finite else np.full_like(target_camera, np.nan)
    s_right, t_right = source_obs[:, 2], target_obs[:, 2]
    if duplicates:
        s_pixels = np.vstack([s_pixels, np.repeat(s_pixels[:1], duplicates, axis=0)])
        t_pixels = np.vstack([t_pixels, np.repeat(t_pixels[:1], duplicates, axis=0)])
        s_points = np.vstack([s_points, np.repeat(s_points[:1], duplicates, axis=0)])
        t_points = np.vstack([t_points, np.repeat(t_points[:1], duplicates, axis=0)])
        s_right = np.r_[s_right, np.repeat(s_right[:1], duplicates)]
        t_right = np.r_[t_right, np.repeat(t_right[:1], duplicates)]
        sid = np.r_[sid, np.repeat(sid[:1], duplicates)]
        tid = np.r_[tid, np.repeat(tid[:1], duplicates)]
    source = SupportedStereoFrame(s_pixels, np.zeros((len(s_pixels), 32), np.uint8), s_points,
                                  s_right, sid, 126, SIZE, CALIB)
    target = SupportedStereoFrame(t_pixels, np.zeros((len(t_pixels), 32), np.uint8), t_points,
                                  t_right, tid, 127, SIZE, CALIB)
    # Source is a non-keyframe transported from map anchor 23; target is keyframe anchor 24.
    source_endpoint = EndpointPose(126, source_pose, 23, anchor0, "tracking", EPOCH)
    target_endpoint = EndpointPose(127, target_pose, 24, target_pose, "tracking", EPOCH)
    return world, source, target, source_endpoint, target_endpoint


def _build(scene, *, pairs=None, fwd=None, rev=None, held=(), **kwargs):
    _, source, target, se, te = scene
    n = len(source.pixels) if pairs is None else len(pairs)
    if pairs is None:
        pairs = np.c_[np.arange(n), np.arange(n)]
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return build_stereo_training_factors(
        source, target, se, te, matrix=K, baseline=B, disparity_offset=OFFSET,
        calibration_identity=CALIB, fit_pairs=pairs,
        forward_inlier_pairs=pairs if fwd is None else np.asarray(fwd, dtype=np.int64).reshape(-1,2),
        reverse_inlier_pairs=pairs if rev is None else np.asarray(rev, dtype=np.int64).reshape(-1,2),
        heldout_pairs=np.asarray(held, dtype=np.int64).reshape(-1,2), **kwargs)


def test_factor_builder_preserves_roles_ownership_and_excludes_held_physical_aliases():
    scene = _scene(n=8, duplicates=2)
    pairs = np.array([[0,0], [8,8], [1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7]])
    factors, report = _build(scene, pairs=pairs, fwd=pairs[:5], rev=pairs[1:7])
    assert report["status"] == "accepted"
    assert len(factors) == 6
    first = factors[0]
    assert first.source_alias_rows == (0, 8, 9)
    assert first.target_alias_rows == (0, 8, 9)
    assert first.forward_role and first.reverse_role
    assert first.covariance_claim is False and first.heldout_validation_claim is False
    with pytest.raises(ValueError, match="invalid_factor_certification_metadata"):
        replace(first, covariance_claim=True)
    # One heldout alias row excludes the entire exact physical source and target group.
    kept, held_report = _build(scene, pairs=pairs, fwd=pairs[:5], rev=pairs[1:7], held=[[9,9]])
    assert len(kept) == 5
    assert held_report["rejections"]["heldout_physical_endpoint"] >= 1
    assert all(f.physical_identity != first.physical_identity for f in kept)
    assert not first.source_pixel.flags.writeable
    payload = first.to_dict()
    assert payload["forward_role"] is True and payload["heldout_validation_claim"] is False


def test_unsupported_feature_id_claim_is_not_treated_as_actual_observation_owner():
    ids = np.full(4, -1, np.int32);ids[0] = 77
    scene = _scene(n=4, lm_ids=ids)
    factors, report = _build(scene)
    assert len(factors) == 4
    assert factors[0].source_landmark_id == -1
    assert factors[0].reused_landmark_id == -1
    assert report["rejections"]["unverified_landmark_claim_ignored"] == 1


def test_existing_target_point_reused_only_with_exact_pixel_and_right_u_certificate():
    ids = np.full(4, -1, np.int32);ids[0] = 77
    scene = _scene(n=4, lm_ids=ids)
    _, _, target, _, _ = scene
    row = {"frame_id": target.frame, "landmark_id": 77,
           "pixel": target.pixels[0], "right_u": target.right_u[0]}
    world, source, target, se, te = scene
    src_point = source.points[0]
    point_world = se.pose[:3,:3] @ src_point + se.pose[:3,3]
    factors, report = _build(scene, existing_observations=[row], existing_landmark_points={77:point_world})
    assert report["status"] == "accepted"
    assert factors[0].target_landmark_id == 77 and factors[0].reused_landmark_id == 77
    assert factors[0].target_has_existing_observation
    bad = dict(row, pixel=target.pixels[0] + np.array([.25, 0.]))
    factors_bad, report_bad = _build(scene, existing_observations=[bad], existing_landmark_points={77:point_world})
    assert not any(f.reused_landmark_id == 77 for f in factors_bad)
    assert report_bad["rejections"]["conflicting_endpoint_landmark_ids"] == 1
    no_point, no_point_report = _build(scene, existing_observations=[row])
    assert not any(f.reused_landmark_id == 77 for f in no_point)
    assert no_point_report["rejections"]["missing_existing_landmark_point"] == 1


def test_conflicting_physical_edges_and_alias_landmark_ids_are_rejected():
    scene = _scene(n=3, duplicates=1)
    # Source physical group 0 is paired to two distinct target pixels.
    pair_rows = np.array([[0,0], [3,1], [1,1], [2,2]])
    factors, report = _build(scene, pairs=pair_rows, fwd=pair_rows, rev=pair_rows)
    assert len(factors) == 1
    assert report["rejections"]["ambiguous_physical_correspondence"] == 3
    # Exact aliases with different map-ID claims cannot be arbitrarily resolved.
    world, source, target, se, te = scene
    source_ids = source.landmark_ids.copy();source_ids[0] = 9;source_ids[3] = 10
    source2 = SupportedStereoFrame(source.pixels, source.descriptors, source.points, source.right_u,
                                   source_ids, source.frame, source.image_size, source.calibration_identity)
    fs, rs = _build((world,source2,target,se,te), pairs=np.array([[0,0],[3,3]]),
                    fwd=np.array([[0,0],[3,3]]),rev=np.array([[0,0],[3,3]]))
    assert fs == () and rs["rejections"]["ambiguous_landmark_identity"] == 1


def test_bad_epoch_calibration_pose_partition_and_cap_fail_closed():
    scene = _scene(n=5)
    factors, report = _build(scene, fit_source="full_supported_reference")
    assert factors == () and report["reason"] == "unusable_training_partition"
    world, source, target, se, te = scene
    stale = EndpointPose(127, te.pose, 24, te.anchor_pose, "tracking", (12, 5))
    factors, report = build_stereo_training_factors(source,target,se,stale,matrix=K,baseline=B,
        disparity_offset=OFFSET,calibration_identity=CALIB,fit_pairs=np.c_[np.arange(5),np.arange(5)],
        forward_inlier_pairs=np.c_[np.arange(5),np.arange(5)],reverse_inlier_pairs=np.c_[np.arange(5),np.arange(5)])
    assert factors == () and report["reason"] == "endpoint_epoch_mismatch"
    factors, report = build_stereo_training_factors(source,target,se,te,matrix=K,baseline=B,
        disparity_offset=OFFSET,calibration_identity="other",fit_pairs=np.c_[np.arange(5),np.arange(5)],
        forward_inlier_pairs=np.c_[np.arange(5),np.arange(5)],reverse_inlier_pairs=np.c_[np.arange(5),np.arange(5)])
    assert factors == () and report["reason"] == "calibration_identity_mismatch"
    _, cap_report = _build(scene, maximum=257)
    assert cap_report["reason"] == "invalid_factor_cap"


def test_out_of_domain_or_nonpositive_disparity_rows_are_rejected():
    scene=_scene(n=4)
    world,source,target,se,te=scene
    target_pixels=target.pixels.copy();target_pixels[0,0]=SIZE[0]+1
    outside=SupportedStereoFrame(target_pixels,target.descriptors,target.points,target.right_u,
                                 target.landmark_ids,target.frame,target.image_size,target.calibration_identity)
    factors,report=build_stereo_training_factors(source,outside,se,te,matrix=K,baseline=B,
        disparity_offset=OFFSET,calibration_identity=CALIB,fit_pairs=np.c_[np.arange(4),np.arange(4)],
        forward_inlier_pairs=np.c_[np.arange(4),np.arange(4)],reverse_inlier_pairs=np.c_[np.arange(4),np.arange(4)])
    assert len(factors)==3 and report["rejections"]["invalid_sensor_geometry"]==1
    bad_right=target.right_u.copy();bad_right[0]=target.pixels[0,0]+1
    nonpositive=SupportedStereoFrame(target.pixels,target.descriptors,target.points,bad_right,
                                     target.landmark_ids,target.frame,target.image_size,target.calibration_identity)
    factors,report=build_stereo_training_factors(source,nonpositive,se,te,matrix=K,baseline=B,
        disparity_offset=OFFSET,calibration_identity=CALIB,fit_pairs=np.c_[np.arange(4),np.arange(4)],
        forward_inlier_pairs=np.c_[np.arange(4),np.arange(4)],reverse_inlier_pairs=np.c_[np.arange(4),np.arange(4)])
    assert len(factors)==3 and report["rejections"]["invalid_sensor_geometry"]==1


def test_cap_is_deterministic_and_at_most_256_rows():
    scene = _scene(n=270)
    factors1, report1 = _build(scene)
    factors2, report2 = _build(scene)
    assert len(factors1) == len(factors2) == 256
    assert [f.physical_identity for f in factors1] == [f.physical_identity for f in factors2]
    assert report1["truncated"] == report2["truncated"] == 14
    assert report1["selection_policy"].startswith("lexicographic_exact_float32")
    assert report1["pose_fit_performed"] is False


def test_analytic_anchor_transport_and_point_jacobians_match_finite_differences():
    factors, _ = _build(_scene(n=8))
    factor = factors[2]
    X = factor.point_initial + np.array([.07, -.04, .09])
    A = factor.source_anchor_pose.copy()
    T = factor.target_anchor_pose.copy()
    analytic = linearize_training_factor(factor, source_anchor_pose=A,target_anchor_pose=T,point_world=X)
    eps = 1e-7
    for anchor, pose, key in ((factor.source_anchor,A,factor.source_anchor),(factor.target_anchor,T,factor.target_anchor)):
        numeric = np.empty((6,6))
        for j in range(6):
            e = np.zeros(6);e[j] = eps
            plus = A.copy() if anchor == factor.source_anchor else T.copy()
            minus = plus.copy()
            plus = plus @ se3_exp(e);minus = minus @ se3_exp(-e)
            if anchor == factor.source_anchor:
                rp=linearize_training_factor(factor,source_anchor_pose=plus,target_anchor_pose=T,point_world=X)["residual"]
                rm=linearize_training_factor(factor,source_anchor_pose=minus,target_anchor_pose=T,point_world=X)["residual"]
            else:
                rp=linearize_training_factor(factor,source_anchor_pose=A,target_anchor_pose=plus,point_world=X)["residual"]
                rm=linearize_training_factor(factor,source_anchor_pose=A,target_anchor_pose=minus,point_world=X)["residual"]
            numeric[:,j]=(rp-rm)/(2*eps)
        assert np.allclose(numeric,analytic["anchor_jacobians"][key],atol=2e-5,rtol=2e-5)
    numeric_point=np.empty((6,3))
    for j in range(3):
        e=np.zeros(3);e[j]=eps
        rp=linearize_training_factor(factor,point_world=X+e)["residual"]
        rm=linearize_training_factor(factor,point_world=X-e)["residual"]
        numeric_point[:,j]=(rp-rm)/(2*eps)
    assert np.allclose(numeric_point,analytic["point_jacobian"],atol=2e-5,rtol=2e-5)
    assert analytic["residual"].shape == (6,) and analytic["covariance_claim"] is False


def test_common_left_gauge_transform_and_shared_anchor_transport_invariance():
    scene = _scene(n=9)
    factors, _ = _build(scene)
    factor = factors[0]
    r0=linearize_training_factor(factor)["residual"]
    G=se3_exp([1.2,-.7,.4,.17,-.21,.08])
    world, source, target, se, te = scene
    se_g=EndpointPose(se.frame_id,G@se.pose,se.anchor_keyframe_id,G@se.anchor_pose,se.status,se.source_epoch)
    te_g=EndpointPose(te.frame_id,G@te.pose,te.anchor_keyframe_id,G@te.anchor_pose,te.status,te.source_epoch)
    world_g=(G[:3,:3]@world.T).T+G[:3,3]
    source_g=SupportedStereoFrame(source.pixels,source.descriptors,source.points,source.right_u,
                                  source.landmark_ids,source.frame,source.image_size,source.calibration_identity)
    target_g=SupportedStereoFrame(target.pixels,target.descriptors,target.points,target.right_u,
                                  target.landmark_ids,target.frame,target.image_size,target.calibration_identity)
    factors_g,_=build_stereo_training_factors(source_g,target_g,se_g,te_g,matrix=K,baseline=B,
        disparity_offset=OFFSET,calibration_identity=CALIB,fit_pairs=np.c_[np.arange(9),np.arange(9)],
        forward_inlier_pairs=np.c_[np.arange(9),np.arange(9)],reverse_inlier_pairs=np.c_[np.arange(9),np.arange(9)])
    f_g=next(x for x in factors_g if x.physical_identity==factor.physical_identity)
    assert np.allclose(f_g.point_initial, G[:3,:3]@factor.point_initial+G[:3,3], atol=1e-9)
    assert np.allclose(linearize_training_factor(f_g)["residual"],r0,atol=2e-5)
    # Two virtual images on one anchor preserve their relative camera transform.
    C1=_pose([.1,0,.01,.02,-.01,.04]);C2=_pose([.4,.02,.03,-.01,.04,.08]);A=_pose([0,.1,.2,.03,.01,-.02])
    Z=np.linalg.inv(A@C1)@(A@C2)
    A2=A@se3_exp([.1,-.2,.03,.04,.02,-.06])
    assert np.allclose(np.linalg.inv(A2@C1)@(A2@C2),Z,atol=1e-10)


def test_shared_map_point_base_rows_are_deduplicated_before_point_schur():
    ids=np.full(5,-1,np.int32);ids[0]=7
    scene=_scene(n=5,lm_ids=ids)
    world,source,target,se,te=scene
    pixel_owner={"frame_id":target.frame,"landmark_id":7,"pixel":target.pixels[0],"right_u":target.right_u[0]}
    point=se.pose[:3,:3]@source.points[0]+se.pose[:3,3]
    factors,report=_build(scene,existing_observations=[pixel_owner],existing_landmark_points={7:point})
    f=factors[0]
    base={"frame_id":target.frame,"landmark_id":7,"anchor_id":te.anchor_keyframe_id,
          "anchor_pose":te.anchor_pose,"camera_pose":te.pose,"pixel":target.pixels[0],
          "right_u":target.right_u[0],"point_world":point}
    one=accumulate_schur_information([f],free_anchor_ids=[23,24])
    duplicated=accumulate_schur_information([f],free_anchor_ids=[23,24],base_observations=[base])
    assert one["raw_observation_rows"]==2 and duplicated["base_observation_rows_supplied"]==1
    assert np.allclose(one["information"],duplicated["information"],atol=1e-7)
    assert duplicated["point_group_count"]==1
    duplicate_factor=accumulate_schur_information([f,f],free_anchor_ids=[23,24])
    assert np.allclose(one["information"],duplicate_factor["information"],atol=1e-7)
    bad=dict(base,right_u=float(base["right_u"])+.5)
    with pytest.raises(ValueError,match="conflicting_duplicate_physical_observation"):
        accumulate_schur_information([f],free_anchor_ids=[23,24],base_observations=[bad])


def test_pose_schur_fixed_boundary_and_rotation_reports_distinguish_relative_from_absolute():
    factors,_=_build(_scene(n=18))
    f=factors[0]
    fixed_source=factor_schur_information(f,free_anchor_ids=[f.target_anchor])
    assert fixed_source["information"].shape==(6,6)
    both=accumulate_schur_information(factors,free_anchor_ids=[23,24])
    summary=motion_rotation_information(both,source_anchor=23,target_anchor=24,
        source_anchor_relative=f.source_anchor_relative,target_anchor_relative=f.target_anchor_relative,
        relative_pose=np.linalg.inv(f.source_pose)@f.target_pose)
    assert summary["reason"] is None
    assert summary["absolute_newest_rotation"]["information"].shape==(3,3)
    assert summary["relative_rotation"]["information"].shape==(3,3)
    assert np.trace(summary["relative_rotation"]["information"]) > 1e-8
    assert np.trace(summary["absolute_newest_rotation"]["information"]) < 1e-6
    assert summary["covariance_claim"] is False
    shared=motion_rotation_information(both,source_anchor=23,target_anchor=23,
        source_anchor_relative=np.eye(4),target_anchor_relative=np.eye(4),relative_pose=np.eye(4))
    assert shared["relative_rotation"] is None


def test_historical_base_rows_and_training_rows_share_one_point_before_schur():
    ids=np.full(4,-1,np.int32);ids[0]=7
    scene=_scene(n=4,lm_ids=ids)
    world,source,target,se,te=scene
    owner={"frame_id":target.frame,"landmark_id":7,"pixel":target.pixels[0],"right_u":target.right_u[0]}
    point=se.pose[:3,:3]@source.points[0]+se.pose[:3,3]
    factors,_=_build(scene,existing_observations=[owner],existing_landmark_points={7:point})
    factor=factors[0]
    old_pose=se3_exp([-.35,.06,.09,-.07,.025,-.04])
    old_obs=_project(old_pose,point)
    old={"frame_id":120,"landmark_id":7,"anchor_id":22,"anchor_pose":old_pose,
         "camera_pose":old_pose,"pixel":old_obs[:2],"right_u":old_obs[2],"point_world":point}
    target_duplicate={"frame_id":target.frame,"landmark_id":7,"anchor_id":24,
         "anchor_pose":te.anchor_pose,"camera_pose":te.pose,"pixel":target.pixels[0],
         "right_u":target.right_u[0],"point_world":point}
    grouped=accumulate_schur_information([factor],free_anchor_ids=[22,23,24],
                                        base_observations=[target_duplicate,old])
    # Independent elimination per input block misses cross-anchor constraints
    # mediated by the one shared nuisance point.
    raw=stf._factor_observation_rows(factor,point_world=point)
    raw_lin=linearize_training_factor(factor,point_world=point)
    Jc_raw=np.zeros((6,18));
    for r,row in enumerate(raw):
        c=[22,23,24].index(row["anchor_id"])
        Jc_raw[3*r:3*r+3,6*c:6*c+6]=row["anchor_jacobian"]
    Hindependent=stf._project_point_nuisance(Jc_raw,raw_lin["point_jacobian"])
    oldrow=stf._base_observation_row(old,factor,{})
    Jc_old=np.zeros((3,18));Jc_old[:,0:6]=oldrow["anchor_jacobian"]
    H_old_separate=stf._project_point_nuisance(Jc_old,oldrow["point_jacobian"])
    # The one-view block has no standalone pose information after eliminating X.
    assert np.linalg.norm(H_old_separate) < 1e-7
    Jc_stacked = np.vstack((Jc_raw, Jc_old))
    Jx_stacked = np.vstack((raw_lin["point_jacobian"], oldrow["point_jacobian"]))
    joint_oracle = stf._project_point_nuisance(Jc_stacked, Jx_stacked)
    assert np.allclose(grouped["information"], joint_oracle, atol=1e-6)
    assert np.linalg.norm(grouped["information"]-Hindependent) > 1e-4
    assert grouped["base_observation_rows_supplied"]==2


def test_broad_training_rows_add_relative_roll_information_under_fixed_pixel_noise():
    def relative_roll_information(scene):
        factors,_=_build(scene)
        h=accumulate_schur_information(factors,free_anchor_ids=[23,24])
        f=factors[0]
        report=motion_rotation_information(h,source_anchor=23,target_anchor=24,
            source_anchor_relative=f.source_anchor_relative,target_anchor_relative=f.target_anchor_relative,
            relative_pose=np.linalg.inv(f.source_pose)@f.target_pose)
        return report["relative_rotation"]["information"][2,2]
    broad=relative_roll_information(_scene(n=30,broad=True))
    narrow=relative_roll_information(_scene(n=30,broad=False))
    assert np.isfinite(broad) and np.isfinite(narrow)
    assert broad > narrow
