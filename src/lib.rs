use nalgebra::{DMatrix, DVector, Isometry3, IsometryMatrix3, Matrix3, Rotation3, Vector3};

type Matrix9<T> = nalgebra::SMatrix<T, 9, 9>;

pub fn solve_ax_xb_andreff(poses: &[(Isometry3<f64>, Isometry3<f64>)]) -> Result<IsometryMatrix3<f64>, String> {
	let mut m: DMatrix<f64> = DMatrix::zeros(poses.len() * 12, 12);
	let mut d: DVector<f64> = DVector::zeros(poses.len() * 12);

	for (i, (a, b)) in poses.iter().enumerate() {
		let rotation_a = *a.rotation.to_rotation_matrix().matrix();
		let rotation_b = *b.rotation.to_rotation_matrix().matrix();
		let translation_a = a.translation.vector;
		let translation_b = b.translation.vector;

		m.fixed_slice_mut::<9, 9>(12 * i, 0)
			.copy_from(&(Matrix9::identity() - rotation_a.kronecker(&rotation_b)));
		m.fixed_slice_mut::<3, 9>(12 * i + 9, 0)
			.copy_from(&Matrix3::identity().kronecker(&translation_b.transpose()));
		m.fixed_slice_mut::<3, 3>(12 * i + 9, 9)
			.copy_from(&(Matrix3::identity() - rotation_a));
		d.fixed_slice_mut::<3, 1>(12 * i + 9, 0).copy_from(&translation_a);
	}

	let svd = nalgebra::SVD::new(m, true, true);
	let x = svd.solve(&d, f64::EPSILON)?;
	let rotation = Matrix3::from_row_slice(&x.as_slice()[..9]);

	let svd = nalgebra::SVD::new(rotation, true, true);
	let rotation = svd.u.unwrap() * svd.v_t.unwrap();
	let translation = x.fixed_rows::<3>(9);

	let rotation = Rotation3::from_matrix(&rotation);
	Ok(IsometryMatrix3 {
		translation: Vector3::from(translation).into(),
		rotation,
	})
}

pub fn solve_ax_xb_andreff_extended(
	poses: &[(Isometry3<f64>, Isometry3<f64>)],
) -> Result<IsometryMatrix3<f64>, String> {
	let mut m: DMatrix<f64> = DMatrix::zeros(poses.len() * 9, 9);
	let mut n: DMatrix<f64> = DMatrix::zeros(poses.len() * 4, 4);

	for (i, (a, b)) in poses.iter().enumerate() {
		let rotation_a = *a.rotation.to_rotation_matrix().matrix();
		let rotation_b = *b.rotation.to_rotation_matrix().matrix();
		let translation_a = a.translation.vector;

		m.fixed_slice_mut::<9, 9>(9 * i, 0)
			.copy_from(&(Matrix9::identity() - rotation_a.kronecker(&rotation_b)));
		n.fixed_slice_mut::<3, 3>(3 * i, 0)
			.copy_from(&(rotation_a - Matrix3::identity()));
		n.fixed_slice_mut::<3, 1>(3 * i, 3).copy_from(&translation_a);
	}

	let svd = nalgebra::SVD::new(m, true, true);
	let v = svd.v_t.unwrap().transpose();

	let mut r_alpha = Matrix3::zeros();
	r_alpha.set_row(0, &v.fixed_slice::<3, 1>(0, 11).transpose());
	r_alpha.set_row(1, &v.fixed_slice::<3, 1>(3, 11).transpose());
	r_alpha.set_row(2, &v.fixed_slice::<3, 1>(6, 11).transpose());
	let det = r_alpha.determinant();
	let alpha = det.abs().powf(4.0 / 3.0) / det;
	let qr = nalgebra::QR::new(r_alpha / alpha);
	let q = qr.q();

	let mut rotation = Matrix3::zeros();
	let r_with_scale = alpha * q.transpose() * r_alpha;
	for i in 0..3 {
		if r_with_scale.diagonal()[i] >= 0.0 {
			rotation.column_mut(i).copy_from(&qr.q().column(i));
		} else {
			rotation.column_mut(i).copy_from(&-qr.q().column(i));
		}
	}
	let rotation = Rotation3::from_matrix(&rotation);

	let mut d: DVector<f64> = DVector::zeros(poses.len() * 3);
	for (i, (_a, b)) in poses.iter().enumerate() {
		d.fixed_rows_mut::<3>(3 * i)
			.copy_from(&(rotation * b.translation.vector));
	}

	let translation: Vector3<_> = n.svd(true, true).solve(&d, f64::EPSILON)?.fixed_rows::<3>(0).into();

	Ok(IsometryMatrix3 {
		translation: translation.into(),
		rotation,
	})
}

pub fn solve_ax_xb_elil(poses: &[(Isometry3<f64>, Isometry3<f64>)]) -> Result<IsometryMatrix3<f64>, String> {
	let mut m: DMatrix<f64> = DMatrix::zeros(poses.len() * 12, 12);

	for (i, (a, b)) in poses.iter().enumerate() {
		let rotation_a = *a.rotation.to_rotation_matrix().matrix();
		let rotation_b = *b.rotation.to_rotation_matrix().matrix();
		let translation_a = a.translation.vector;
		let translation_b = b.translation.vector;

		m.fixed_slice_mut::<9, 9>(12 * i, 0)
			.copy_from(&(Matrix9::identity() - rotation_a.kronecker(&rotation_b)));

		let skew = skew(translation_a);
		m.fixed_slice_mut::<3, 9>(12 * i + 9, 0)
			.copy_from(&skew.kronecker(&translation_b.transpose()));
		m.fixed_slice_mut::<3, 3>(12 * i + 9, 9)
			.copy_from(&(skew - skew * rotation_a));
	}

	let svd = m.svd(true, true);
	let v = svd.v_t.unwrap().transpose();

	let mut r_alpha = Matrix3::zeros();
	r_alpha.set_row(0, &v.fixed_slice::<3, 1>(0, 11).transpose());
	r_alpha.set_row(1, &v.fixed_slice::<3, 1>(3, 11).transpose());
	r_alpha.set_row(2, &v.fixed_slice::<3, 1>(6, 11).transpose());
	let det = r_alpha.determinant();
	let alpha = det.abs().powf(4.0 / 3.0);
	let qr = (r_alpha / alpha).qr();
	let q = qr.q();

	let mut rotation = Matrix3::zeros();
	let r_with_scale = alpha * q.transpose() * r_alpha;
	for i in 0..3 {
		if r_with_scale.diagonal()[i] >= 0.0 {
			rotation.column_mut(i).copy_from(&qr.q().column(i));
		} else {
			rotation.column_mut(i).copy_from(&-qr.q().column(i));
		}
	}
	let rotation = Rotation3::from_matrix(&rotation);

	let translation = v.fixed_slice::<3, 1>(9, 11) / alpha;

	Ok(IsometryMatrix3 {
		translation: translation.into(),
		rotation,
	})
}

fn skew(u: Vector3<f64>) -> Matrix3<f64> {
	nalgebra::matrix![
		 0.0,  u.z, -u.y;
		-u.z,  0.0,  u.x;
		 u.y, -u.x, 0.0;
	]
}
