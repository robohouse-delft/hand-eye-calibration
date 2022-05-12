use assert2::let_assert;
use hand_eye_calibration::solve_ax_xb_andreff;
use nalgebra::{Isometry3, Vector3, Translation3, Rotation3, Unit, UnitQuaternion};

struct TestSet {
	samples: Vec<(Isometry3<f64>, Isometry3<f64>)>,
	camera_to_ee: Isometry3<f64>,
	pattern_to_robot: Isometry3<f64>
}

fn pose(x: f64, y: f64, z: f64, rx: f64, ry: f64, rz: f64) -> Isometry3<f64> {
	Translation3::new(x, y, z) * UnitQuaternion::new(Vector3::new(rx, ry, rz))
}

fn rot_x(degrees: f64) -> UnitQuaternion<f64> {
	let axis = Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0));
	UnitQuaternion::from_axis_angle(&axis, degrees.to_radians())
}

fn rot_y(degrees: f64) -> UnitQuaternion<f64> {
	let axis = Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0));
	UnitQuaternion::from_axis_angle(&axis, degrees.to_radians())
}

fn rot_z(degrees: f64) -> UnitQuaternion<f64> {
	let axis = Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0));
	UnitQuaternion::from_axis_angle(&axis, degrees.to_radians())
}

fn test_poses() -> TestSet {
	let camera_to_ee = pose(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
	let pattern_to_robot = pose(0.7, 0.8, 0.9, 1.0, 1.1, 1.2);

	let mut samples = Vec::new();
	for rx in -1..=1 {
		let rx = rx as f64 * 10.0;
		for ry in -1..=1 {
			let ry = ry as f64 * 10.0;
			let camera_to_pattern = rot_x(rx) * rot_y(ry) * Translation3::new(0.0, 0.0, 1.0) * rot_x(180.0);
			let camera_to_robot = pattern_to_robot * camera_to_pattern;
			samples.push((camera_to_pattern.inverse(), camera_to_robot * camera_to_ee.inverse()));
		}
	}

	TestSet {
		samples,
		camera_to_ee,
		pattern_to_robot,
	}
}

#[test]
fn andreff() {
	let test_data = test_poses();
	let_assert!(Ok(result) = hand_eye_calibration::solve_ax_xb_elil(&test_data.samples));
}
