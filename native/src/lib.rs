use numpy::{
    IntoPyArray,

    PyArray1,

    PyReadonlyArray1,

};

use pyo3::prelude::*;
use rayon::prelude::*;





#[pyfunction]

fn lower_scaffold(

    py: Python<'_>,

    ids: PyReadonlyArray1<'_, i64>,

    x: PyReadonlyArray1<'_, f64>,

    y: PyReadonlyArray1<'_, f64>,

    z: PyReadonlyArray1<'_, f64>,

    cell_m: f64,

) -> PyResult<Py<PyArray1<i64>>> {



    let ids = ids.as_slice()?;

    let x = x.as_slice()?;

    let y = y.as_slice()?;

    let z = z.as_slice()?;



    let n = ids.len();



    if x.len() != n ||

       y.len() != n ||

       z.len() != n

    {

        return Err(

            pyo3::exceptions::PyValueError::new_err(

                "ids, x, y and z must have identical lengths"

            )

        );

    }



    if n == 0 {

        return Ok(

            PyArray1::from_vec(

                py,

                Vec::<i64>::new()

            ).unbind()

        );

    }



    if !(cell_m.is_finite() && cell_m > 0.0) {

        return Err(

            pyo3::exceptions::PyValueError::new_err(

                "cell_m must be finite and > 0"

            )

        );

    }



    // --------------------------------------------------------

    // Exact FAST-GC reference origins

    // --------------------------------------------------------



    let mut x0 = x[0];

    let mut y0 = y[0];



    for i in 1..n {



        if x[i] < x0 {

            x0 = x[i];

        }



        if y[i] < y0 {

            y0 = y[i];

        }

    }



    // --------------------------------------------------------

    // Exact integer cells

    // --------------------------------------------------------



    let mut ix =

        Vec::<i64>::with_capacity(n);



    let mut iy =

        Vec::<i64>::with_capacity(n);



    let mut max_ix: i64 = 0;



    for i in 0..n {



        let fx =

            ((x[i] - x0) / cell_m).floor();



        let fy =

            ((y[i] - y0) / cell_m).floor();



        if !fx.is_finite() ||

           !fy.is_finite()

        {

            return Err(

                pyo3::exceptions::PyValueError::new_err(

                    "non-finite scaffold coordinate"

                )

            );

        }



        let ixi = fx as i64;

        let iyi = fy as i64;



        if ixi > max_ix {

            max_ix = ixi;

        }



        ix.push(ixi);

        iy.push(iyi);

    }



    let nx = max_ix + 1;



    // --------------------------------------------------------

    // Equivalent to:

    //

    // key = iy * nx + ix

    // order = np.argsort(key, kind="mergesort")

    //

    // Rust stable sort preserves original input order when

    // scalar keys are equal.

    // --------------------------------------------------------



    let mut rows =

        Vec::<(i64, usize)>::with_capacity(n);



    for i in 0..n {



        let key = iy[i]

            .checked_mul(nx)

            .and_then(

                |v| v.checked_add(ix[i])

            )

            .ok_or_else(

                || {

                    pyo3::exceptions::PyOverflowError::new_err(

                        "FAST-GC scaffold key overflow"

                    )

                }

            )?;



        rows.push(

            (

                key,

                i,

            )

        );

    }



    rows.sort_by(

        |a, b| a.0.cmp(&b.0)

    );



    // --------------------------------------------------------

    // Exact FAST-GC group semantics.

    //

    // Strict < preserves first-minimum behavior of np.argmin.

    // --------------------------------------------------------



    let mut selected =

        Vec::<i64>::new();



    let mut start = 0usize;



    while start < n {



        let key = rows[start].0;



        let mut end = start + 1;



        while end < n &&

              rows[end].0 == key

        {

            end += 1;

        }



        let mut best_local =

            rows[start].1;



        let mut best_z =

            z[best_local];



        for j in (start + 1)..end {



            let local =

                rows[j].1;



            let zj =

                z[local];



            if zj < best_z {



                best_z =

                    zj;



                best_local =

                    local;

            }

        }



        selected.push(

            ids[best_local]

        );



        start = end;

    }



    Ok(

        PyArray1::from_vec(

            py,

            selected

        ).unbind()

    )

}







#[pyfunction]

fn robust_stats_mask(

    py: Python<'_>,

    residuals: PyReadonlyArray1<'_, f64>,

    mad_floor: f64,

    lower_mult: f64,

    upper_mult: f64,

) -> PyResult<(

    f64,

    f64,

    f64,

    Py<PyArray1<bool>>

)> {



    let r = residuals.as_slice()?;



    let n = r.len();



    if n == 0 {

        return Err(

            pyo3::exceptions::PyValueError::new_err(

                "residual array must not be empty"

            )

        );

    }



    if !mad_floor.is_finite() ||

       !lower_mult.is_finite() ||

       !upper_mult.is_finite()

    {

        return Err(

            pyo3::exceptions::PyValueError::new_err(

                "robust-stat parameters must be finite"

            )

        );

    }



    // --------------------------------------------------------

    // Production integration will only use finite residual

    // vectors after the existing Python finite filtering.

    //

    // Reject nonfinite input rather than silently changing

    // NumPy semantics.

    // --------------------------------------------------------



    for &v in r {



        if !v.is_finite() {



            return Err(

                pyo3::exceptions::PyValueError::new_err(

                    "residuals must be finite"

                )

            );

        }

    }



    // --------------------------------------------------------

    // NumPy-compatible median definition for finite f64 data.

    // --------------------------------------------------------



    fn median_finite(

        values: &[f64]

    ) -> f64 {



        let mut work =

            values.to_vec();



        work.sort_by(

            |a, b|

            a.partial_cmp(b).unwrap()

        );



        let n = work.len();



        if n % 2 == 1 {



            work[n / 2]



        } else {



            let a =

                work[(n / 2) - 1];



            let b =

                work[n / 2];



            (a + b) / 2.0

        }

    }



    let med =

        median_finite(r);



    let mut abs_dev =

        Vec::<f64>::with_capacity(n);



    for &v in r {



        abs_dev.push(

            (v - med).abs()

        );

    }



    let mad =

        median_finite(

            &abs_dev

        );



    let scaled =

        1.4826_f64 * mad;



    let sigma =

        if scaled > mad_floor {

            scaled

        } else {

            mad_floor

        };



    let lo =

        med - lower_mult * sigma;



    let hi =

        med + upper_mult * sigma;



    let mut mask =

        Vec::<bool>::with_capacity(n);



    for &v in r {



        mask.push(

            v >= lo &&

            v <= hi

        );

    }



    Ok(

        (

            med,

            mad,

            sigma,

            PyArray1::from_vec(

                py,

                mask

            ).unbind(),

        )

    )

}









fn median_finite_local(

    values: &[f64]

) -> f64 {



    let mut work =

        values.to_vec();



    work.sort_by(

        |a, b|

        a.partial_cmp(b).unwrap()

    );



    let n = work.len();



    if n % 2 == 1 {



        work[n / 2]



    } else {



        let a =

            work[(n / 2) - 1];



        let b =

            work[n / 2];



        (a + b) / 2.0

    }

}





fn solve_3x3_local(

    mut a: [[f64; 3]; 3],

    mut b: [f64; 3],

) -> Option<[f64; 3]> {



    // Gaussian elimination with partial pivoting.

    //

    // EXPERIMENTAL ONLY:

    // this is not claimed to be equivalent to LAPACK lstsq.



    for col in 0..3 {



        let mut pivot = col;

        let mut pivot_abs =

            a[col][col].abs();



        for row in (col + 1)..3 {



            let v =

                a[row][col].abs();



            if v > pivot_abs {

                pivot = row;

                pivot_abs = v;

            }

        }



        if !pivot_abs.is_finite()

            || pivot_abs <= 1.0e-15

        {

            return None;

        }



        if pivot != col {



            a.swap(

                pivot,

                col,

            );



            b.swap(

                pivot,

                col,

            );

        }



        let diag =

            a[col][col];



        for row in (col + 1)..3 {



            let factor =

                a[row][col]

                / diag;



            a[row][col] = 0.0;



            for j in (col + 1)..3 {



                a[row][j] -=

                    factor

                    * a[col][j];

            }



            b[row] -=

                factor

                * b[col];

        }

    }



    let mut x =

        [0.0_f64; 3];



    for row in (0..3).rev() {



        let mut rhs =

            b[row];



        for j in (row + 1)..3 {



            rhs -=

                a[row][j]

                * x[j];

        }



        let diag =

            a[row][row];



        if !diag.is_finite()

            || diag.abs() <= 1.0e-15

        {

            return None;

        }



        x[row] =

            rhs / diag;

    }



    if x.iter().all(

        |v| v.is_finite()

    ) {



        Some(x)



    } else {



        None

    }

}





#[pyfunction]

fn surface_robust_plane_experimental(

    x: PyReadonlyArray1<'_, f64>,

    y: PyReadonlyArray1<'_, f64>,

    z: PyReadonlyArray1<'_, f64>,

    asymmetric_high: bool,

) -> PyResult<Option<(

    f64,

    f64,

    f64,

    f64,

    f64,

    f64,

    Vec<bool>

)>> {



    let x =

        x.as_slice()?;



    let y =

        y.as_slice()?;



    let z =

        z.as_slice()?;



    let n =

        x.len();



    if y.len() != n

        || z.len() != n

    {

        return Err(

            pyo3::exceptions::PyValueError::new_err(

                "x, y and z must have equal length"

            )

        );

    }



    // Exact reference starts with finite intersection.

    let mut keep =

        Vec::<bool>::with_capacity(n);



    for i in 0..n {



        keep.push(

            x[i].is_finite()

            && y[i].is_finite()

            && z[i].is_finite()

        );

    }



    let mut nkeep =

        keep.iter()

        .filter(|&&v| v)

        .count();



    if nkeep < 3 {

        return Ok(None);

    }



    let mut xkeep =

        Vec::<f64>::with_capacity(nkeep);



    let mut ykeep =

        Vec::<f64>::with_capacity(nkeep);



    for i in 0..n {



        if keep[i] {



            xkeep.push(x[i]);

            ykeep.push(y[i]);

        }

    }



    let xc =

        median_finite_local(

            &xkeep

        );



    let yc =

        median_finite_local(

            &ykeep

        );



    let mut xx =

        Vec::<f64>::with_capacity(n);



    let mut yy =

        Vec::<f64>::with_capacity(n);



    for i in 0..n {



        xx.push(

            x[i] - xc

        );



        yy.push(

            y[i] - yc

        );

    }



    let mut coef:

        Option<[f64; 3]> =

        None;



    for _ in 0..5 {



        nkeep =

            keep.iter()

            .filter(|&&v| v)

            .count();



        if nkeep < 3 {

            return Ok(None);

        }



        // A = [xx, yy, 1].

        //

        // Solve normal equations:

        //

        // (A^T A)c = A^T z

        //

        // Experimental equivalence candidate only.



        let mut sxx = 0.0_f64;

        let mut syy = 0.0_f64;

        let mut sxy = 0.0_f64;

        let mut sx  = 0.0_f64;

        let mut sy  = 0.0_f64;



        let mut sxz = 0.0_f64;

        let mut syz = 0.0_f64;

        let mut sz  = 0.0_f64;



        for i in 0..n {



            if !keep[i] {

                continue;

            }



            let xi =

                xx[i];



            let yi =

                yy[i];



            let zi =

                z[i];



            sxx += xi * xi;

            syy += yi * yi;

            sxy += xi * yi;



            sx += xi;

            sy += yi;



            sxz += xi * zi;

            syz += yi * zi;

            sz += zi;

        }



        let ata = [

            [

                sxx,

                sxy,

                sx,

            ],

            [

                sxy,

                syy,

                sy,

            ],

            [

                sx,

                sy,

                nkeep as f64,

            ],

        ];



        let atz = [

            sxz,

            syz,

            sz,

        ];



        let current =

            match solve_3x3_local(

                ata,

                atz,

            ) {



                Some(v) => v,



                None => {

                    return Ok(None);

                }

            };



        coef =

            Some(current);



        let mut residuals =

            Vec::<f64>::with_capacity(n);



        for i in 0..n {



            residuals.push(

                z[i]

                - (

                    current[0] * xx[i]

                    + current[1] * yy[i]

                    + current[2]

                )

            );

        }



        let mut rkeep =

            Vec::<f64>::with_capacity(nkeep);



        for i in 0..n {



            if keep[i] {



                rkeep.push(

                    residuals[i]

                );

            }

        }



        let med =

            median_finite_local(

                &rkeep

            );



        let mut dev =

            Vec::<f64>::with_capacity(

                rkeep.len()

            );



        for &r in &rkeep {



            dev.push(

                (r - med).abs()

            );

        }



        let mad =

            1.4826_f64

            * median_finite_local(

                &dev

            );



        let sig =

            if mad > 0.025_f64 {

                mad

            } else {

                0.025_f64

            };



        let mut next =

            vec![false; n];



        if asymmetric_high {



            let lo =

                med

                - 4.0_f64

                * sig;



            let hi =

                med

                + 2.25_f64

                * sig;



            for i in 0..n {



                next[i] =

                    keep[i]

                    && residuals[i] >= lo

                    && residuals[i] <= hi;

            }



        } else {



            let lim =

                3.25_f64

                * sig;



            for i in 0..n {



                next[i] =

                    keep[i]

                    && (

                        residuals[i]

                        - med

                    ).abs() <= lim;

            }

        }



        let nnext =

            next.iter()

            .filter(|&&v| v)

            .count();



        // Exact Python control flow:

        //

        // if nk.sum()<3 or np.array_equal(nk,keep): break

        //

        // In either case keep is NOT replaced.



        if nnext < 3

            || next == keep

        {

            break;

        }



        keep =

            next;

    }



    let c =

        match coef {



            Some(v) => v,



            None => {

                return Ok(None);

            }

        };



    // Exact reference computes RMSE using current keep,

    // but DOES NOT perform a final coefficient refit.



    let mut sse =

        0.0_f64;



    let mut count =

        0_usize;



    for i in 0..n {



        if !keep[i] {

            continue;

        }



        let pred =

            c[0] * xx[i]

            + c[1] * yy[i]

            + c[2];



        let d =

            z[i] - pred;



        sse +=

            d * d;



        count += 1;

    }



    if count == 0 {

        return Ok(None);

    }



    let rmse =

        (

            sse

            / count as f64

        ).sqrt();



    Ok(

        Some(

            (

                c[0],

                c[1],

                c[2],

                xc,

                yc,

                rmse,

                keep,

            )

        )

    )

}









// FASTGC_KERNEL4_FUSED_SURFACE_PLANE_ITERATION_V1
//
// Experimental only.
//
// np.linalg.lstsq remains in Python and remains the
// scientific reference solver.
//
// This function performs only:
//   residual
//   median
//   MAD
//   clipping
//   next robust keep mask

#[pyfunction]
fn surface_plane_iteration<'py>(
    py: Python<'py>,
    xx: PyReadonlyArray1<'py, f64>,
    yy: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    keep: PyReadonlyArray1<'py, bool>,
    coef: PyReadonlyArray1<'py, f64>,
    asymmetric_high: bool,
) -> PyResult<Bound<'py, PyArray1<bool>>> {

    let xx = xx.as_slice()?;
    let yy = yy.as_slice()?;
    let z = z.as_slice()?;
    let keep = keep.as_slice()?;
    let coef = coef.as_slice()?;

    let n = xx.len();

    if yy.len() != n
        || z.len() != n
        || keep.len() != n
    {
        return Err(
            pyo3::exceptions::PyValueError::new_err(
                "xx, yy, z, keep length mismatch"
            )
        );
    }

    if coef.len() != 3 {
        return Err(
            pyo3::exceptions::PyValueError::new_err(
                "coef must have length 3"
            )
        );
    }

    let a = coef[0];
    let b = coef[1];
    let c = coef[2];

    let mut residual =
        Vec::<f64>::with_capacity(n);

    let mut selected =
        Vec::<f64>::new();

    for i in 0..n {

        let r =
            z[i]
            - (
                a * xx[i]
                + b * yy[i]
                + c
            );

        residual.push(r);

        if keep[i] {
            selected.push(r);
        }
    }

    if selected.is_empty() {
        return Err(
            pyo3::exceptions::PyValueError::new_err(
                "keep contains no selected values"
            )
        );
    }

    fn median_local(vin: &[f64]) -> f64 {

        let mut v = vin.to_vec();

        v.sort_by(
            |a, b|
            a.partial_cmp(b).unwrap()
        );

        let n = v.len();

        if n % 2 == 1 {
            v[n / 2]
        } else {
            (
                v[n / 2 - 1]
                + v[n / 2]
            ) * 0.5
        }
    }

    let med =
        median_local(&selected);

    let mut deviations =
        Vec::<f64>::with_capacity(
            selected.len()
        );

    for value in selected.iter() {
        deviations.push(
            (*value - med).abs()
        );
    }

    let mad =
        1.4826_f64
        * median_local(&deviations);

    let sig =
        if mad > 0.025_f64 {
            mad
        } else {
            0.025_f64
        };

    let mut next =
        Vec::<bool>::with_capacity(n);

    if asymmetric_high {

        let lo =
            med - 4.0_f64 * sig;

        let hi =
            med + 2.25_f64 * sig;

        for i in 0..n {

            next.push(
                keep[i]
                && residual[i] >= lo
                && residual[i] <= hi
            );
        }

    } else {

        let lim =
            3.25_f64 * sig;

        for i in 0..n {

            next.push(
                keep[i]
                && (
                    residual[i] - med
                ).abs() <= lim
            );
        }
    }

    Ok(
        next.into_pyarray(py)
    )
}


// ============================================================

// ============================================================
// FAST-GC TLS Kernel #1A
//
// Exact iteration primitive for tls_final_ground_vote._robust_plane.
//
// Scientific contract:
//   dx = px - cx
//   dy = py - cy
//   pred = beta[0]*dx + beta[1]*dy + beta[2]
//   scale = sqrt(1 + beta[0]^2 + beta[1]^2)
//   nr = (pz - pred) / scale
//   med = median(nr[keep])
//   mad = max(median(abs(nr[keep] - med)), mad_floor)
//   sigma = 1.4826 * mad
//   new_keep = abs(nr - med) <= robust_z * sigma
//
// IMPORTANT:
// - new_keep is evaluated over ALL points.
// - The previous keep mask is used only to estimate median/MAD.
// - NumPy np.linalg.lstsq remains in Python.
// ============================================================

#[pyfunction]
fn tls_final_plane_iteration<'py>(
    py: Python<'py>,
    px: PyReadonlyArray1<'py, f64>,
    py_coord: PyReadonlyArray1<'py, f64>,
    pz: PyReadonlyArray1<'py, f64>,
    keep: PyReadonlyArray1<'py, bool>,
    beta: PyReadonlyArray1<'py, f64>,
    cx: f64,
    cy: f64,
    robust_z: f64,
    mad_floor: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {

    let px = px.as_slice()?;
    let py_coord = py_coord.as_slice()?;
    let pz = pz.as_slice()?;
    let keep = keep.as_slice()?;
    let beta = beta.as_slice()?;

    let n = px.len();

    if py_coord.len() != n
        || pz.len() != n
        || keep.len() != n
    {
        return Err(
            pyo3::exceptions::PyValueError::new_err(
                "px, py, pz, keep length mismatch"
            )
        );
    }

    if beta.len() != 3 {
        return Err(
            pyo3::exceptions::PyValueError::new_err(
                "beta must have length 3"
            )
        );
    }

    let a = beta[0];
    let b = beta[1];
    let c = beta[2];

    let scale =
        (1.0_f64 + a * a + b * b).sqrt();

    let mut nr =
        Vec::<f64>::with_capacity(n);

    let mut selected =
        Vec::<f64>::new();

    for i in 0..n {
        let dx = px[i] - cx;
        let dy = py_coord[i] - cy;

        let pred =
            a * dx
            + b * dy
            + c;

        let r =
            (pz[i] - pred) / scale;

        nr.push(r);

        if keep[i] {
            selected.push(r);
        }
    }

    if selected.is_empty() {
        return Err(
            pyo3::exceptions::PyValueError::new_err(
                "keep contains no selected values"
            )
        );
    }

    fn median_local(vin: &[f64]) -> f64 {
        let mut v = vin.to_vec();

        v.sort_by(
            |a, b|
            a.partial_cmp(b).unwrap()
        );

        let n = v.len();

        if n % 2 == 1 {
            v[n / 2]
        } else {
            (
                v[n / 2 - 1]
                + v[n / 2]
            ) * 0.5
        }
    }

    let med =
        median_local(&selected);

    let mut deviations =
        Vec::<f64>::with_capacity(
            selected.len()
        );

    for value in selected.iter() {
        deviations.push(
            (*value - med).abs()
        );
    }

    let raw_mad =
        median_local(&deviations);

    let mad =
        if raw_mad > mad_floor {
            raw_mad
        } else {
            mad_floor
        };

    let sigma =
        1.4826_f64 * mad;

    let limit =
        robust_z * sigma;

    let mut next =
        Vec::<bool>::with_capacity(n);

    for i in 0..n {
        next.push(
            (nr[i] - med).abs()
            <= limit
        );
    }

    Ok(
        next.into_pyarray(py)
    )
}


// FAST-GC Kernel #5
// ULS hard-airborne terrain robust-plane iteration.
//
// Scientific contract:
// residual = z - (a*xx + b*yy + c)
// median over residual[keep]
// sigma = max(1.4826 * MAD, 0.025)
// clipping = [median - 4.0*sigma, median + 1.8*sigma]
//
// np.linalg.lstsq remains in the Python reference path.
// ============================================================

#[pyfunction]
fn uls_terrain_plane_iteration<'py>(
    py: Python<'py>,
    xx: PyReadonlyArray1<'py, f64>,
    yy: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    finite: PyReadonlyArray1<'py, bool>,
    keep: PyReadonlyArray1<'py, bool>,
    coef: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {

    let xx = xx.as_slice()?;
    let yy = yy.as_slice()?;
    let z = z.as_slice()?;
    let finite = finite.as_slice()?;
    let keep = keep.as_slice()?;
    let coef = coef.as_slice()?;

    let n = xx.len();

    if yy.len() != n
        || z.len() != n
        || finite.len() != n
        || keep.len() != n
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "xx, yy, z, keep length mismatch"
        ));
    }

    if coef.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "coef must have length 3"
        ));
    }

    let a = coef[0];
    let b = coef[1];
    let c = coef[2];

    let mut residual = Vec::<f64>::with_capacity(n);
    let mut selected = Vec::<f64>::new();

    for i in 0..n {
        let r = z[i] - (a * xx[i] + b * yy[i] + c);
        residual.push(r);

        if keep[i] {
            selected.push(r);
        }
    }

    if selected.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "keep contains no selected values"
        ));
    }

    fn median_local(vin: &[f64]) -> f64 {
        let mut v = vin.to_vec();

        v.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = v.len();

        if n % 2 == 1 {
            v[n / 2]
        } else {
            (v[n / 2 - 1] + v[n / 2]) * 0.5
        }
    }

    let med = median_local(&selected);

    let mut deviations =
        Vec::<f64>::with_capacity(selected.len());

    for value in selected.iter() {
        deviations.push((*value - med).abs());
    }

    let scaled_mad =
        1.4826_f64 * median_local(&deviations);

    let sig =
        if scaled_mad > 0.025_f64 {
            scaled_mad
        } else {
            0.025_f64
        };

    let lo = med - 4.0_f64 * sig;
    let hi = med + 1.8_f64 * sig;

    let mut next = Vec::<bool>::with_capacity(n);

    for i in 0..n {
        next.push(
            finite[i]
            && residual[i] >= lo
            && residual[i] <= hi
        );
    }

    Ok(next.into_pyarray(py))
}



#[pyfunction]
fn airborne_prepare_support<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    ground: PyReadonlyArray1<'py, bool>,
    nbr: PyReadonlyArray1<'py, i64>,
    point_id: i64,
    exclusion_radius_m: f64,
    scaffold_cell_m: f64,
    min_support_points: usize,
    min_support_sectors: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, usize)> {
    let x = x.as_slice()?;
    let y = y.as_slice()?;
    let z = z.as_slice()?;
    let ground = ground.as_slice()?;
    let nbr = nbr.as_slice()?;

    let n = x.len();

    if y.len() != n || z.len() != n || ground.len() != n {
        return Err(
            pyo3::exceptions::PyValueError::new_err(
                "x, y, z, ground length mismatch"
            )
        );
    }

    if point_id < 0 || point_id as usize >= n {
        return Err(
            pyo3::exceptions::PyValueError::new_err(
                "point_id out of range"
            )
        );
    }

    let pid = point_id as usize;
    let px = x[pid];
    let py0 = y[pid];

    // Exact logical equivalent of:
    //
    // nbr = nbr[result[nbr]]
    // nbr = nbr[nbr != point_id]
    // d = np.hypot(...)
    // support = nbr[d >= exclusion_radius_m]
    //
    let mut support: Vec<i64> = Vec::with_capacity(nbr.len());

    for &raw_id in nbr.iter() {
        if raw_id < 0 || raw_id as usize >= n {
            return Err(
                pyo3::exceptions::PyValueError::new_err(
                    "neighbor id out of range"
                )
            );
        }

        let id = raw_id as usize;

        if !ground[id] || id == pid {
            continue;
        }

        let dx = x[id] - px;
        let dy = y[id] - py0;

        let d = dx.hypot(dy);

        if d >= exclusion_radius_m {
            support.push(raw_id);
        }
    }

    if support.len() < min_support_points {
        let empty: Vec<i64> = Vec::new();
        return Ok((
            PyArray1::from_vec(py, empty),
            0,
        ));
    }

    // Exact Kernel #1 scaffold semantics:
    // minima are computed over support in its current order.
    let mut x0 = f64::INFINITY;
    let mut y0 = f64::INFINITY;

    for &raw_id in support.iter() {
        let id = raw_id as usize;

        if x[id] < x0 {
            x0 = x[id];
        }

        if y[id] < y0 {
            y0 = y[id];
        }
    }

    let mut cells: Vec<(i64, usize, i64)> =
        Vec::with_capacity(support.len());

    let mut max_ix: i64 = 0;

    for (position, &raw_id) in support.iter().enumerate() {
        let id = raw_id as usize;

        let ix = ((x[id] - x0) / scaffold_cell_m).floor() as i64;
        let iy = ((y[id] - y0) / scaffold_cell_m).floor() as i64;

        if ix > max_ix {
            max_ix = ix;
        }

        // Store ix temporarily. Key is formed after nx is known.
        cells.push((ix, position, iy));
    }

    let nx = max_ix + 1;

    let mut keyed: Vec<(i64, usize, i64)> =
        Vec::with_capacity(cells.len());

    for (ix, position, iy) in cells.into_iter() {
        keyed.push((
            iy * nx + ix,
            position,
            support[position],
        ));
    }

    // Rust stable sort preserves support ordering for equal cell keys,
    // matching NumPy argsort(kind="mergesort").
    keyed.sort_by(|a, b| a.0.cmp(&b.0));

    let mut scaffold: Vec<i64> = Vec::new();

    let mut start = 0usize;

    while start < keyed.len() {
        let key = keyed[start].0;
        let mut end = start + 1;

        while end < keyed.len() && keyed[end].0 == key {
            end += 1;
        }

        // np.argmin returns the first minimum.
        let mut best = keyed[start].2;
        let mut best_z = z[best as usize];

        for j in (start + 1)..end {
            let candidate = keyed[j].2;
            let candidate_z = z[candidate as usize];

            if candidate_z < best_z {
                best = candidate;
                best_z = candidate_z;
            }
        }

        scaffold.push(best);
        start = end;
    }

    if scaffold.len() < min_support_points {
        let empty: Vec<i64> = Vec::new();
        return Ok((
            PyArray1::from_vec(py, empty),
            0,
        ));
    }

    // Exact _sector_ids(..., 8) / unique-count semantics.
    let two_pi = 2.0_f64 * std::f64::consts::PI;
    let width = two_pi / 8.0_f64;

    let mut seen = [false; 8];

    for &raw_id in scaffold.iter() {
        let id = raw_id as usize;

        let dx = x[id] - px;
        let dy = y[id] - py0;

        let mut angle = dy.atan2(dx) + two_pi;
        angle = angle % two_pi;

        let sector = (angle / width).floor() as usize;

        if sector < 8 {
            seen[sector] = true;
        }
    }

    let sector_count =
        seen.iter().filter(|&&v| v).count();

    if sector_count < min_support_sectors {
        let empty: Vec<i64> = Vec::new();
        return Ok((
            PyArray1::from_vec(py, empty),
            sector_count,
        ));
    }

    Ok((
        PyArray1::from_vec(py, scaffold),
        sector_count,
    ))
}


#[pyfunction]
fn airborne_prepare_support_multi<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    ground: PyReadonlyArray1<'py, bool>,
    nbr_max: PyReadonlyArray1<'py, i64>,
    point_id: i64,
    radii: PyReadonlyArray1<'py, f64>,
    exclusion_radius_m: f64,
    scaffold_cell_m: f64,
    min_support_points: usize,
    min_support_sectors: usize,
) -> PyResult<Vec<(Bound<'py, PyArray1<i64>>, usize)>> {
    let x = x.as_slice()?;
    let y = y.as_slice()?;
    let z = z.as_slice()?;
    let ground = ground.as_slice()?;
    let nbr_max = nbr_max.as_slice()?;
    let radii = radii.as_slice()?;

    let n = x.len();

    if y.len() != n || z.len() != n || ground.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x, y, z, ground length mismatch"
        ));
    }

    if point_id < 0 || point_id as usize >= n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "point_id out of range"
        ));
    }

    let pid = point_id as usize;
    let px = x[pid];
    let py0 = y[pid];

    /*
    Kernel #6B4.

    nbr_max is the ORDERED result of the existing scalar SciPy
    query_ball_point at max(radii).

    Each radius-specific neighborhood is derived by scanning nbr_max
    in that exact order and retaining d <= radius.

    The remainder deliberately reproduces Kernel #6A semantics
    independently for every radius.
    */

    let mut outputs = Vec::with_capacity(radii.len());

    for &radius in radii.iter() {
        if !radius.is_finite() || radius < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "radii must be finite and non-negative"
            ));
        }

        let mut support: Vec<i64> = Vec::new();

        for &raw_id in nbr_max.iter() {
            if raw_id < 0 || raw_id as usize >= n {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "neighbor id out of range"
                ));
            }

            let id = raw_id as usize;

            let dx = x[id] - px;
            let dy = y[id] - py0;
            let d = dx.hypot(dy);

            // Exact radius-subset boundary established by #6B3.
            if d > radius {
                continue;
            }

            if !ground[id] || id == pid {
                continue;
            }

            if d >= exclusion_radius_m {
                support.push(raw_id);
            }
        }

        if support.len() < min_support_points {
            outputs.push((PyArray1::from_vec(py, Vec::new()), 0));
            continue;
        }

        let mut x0 = f64::INFINITY;
        let mut y0 = f64::INFINITY;

        for &raw_id in support.iter() {
            let id = raw_id as usize;
            if x[id] < x0 { x0 = x[id]; }
            if y[id] < y0 { y0 = y[id]; }
        }

        let mut cells: Vec<(i64, usize, i64)> =
            Vec::with_capacity(support.len());

        let mut max_ix: i64 = 0;

        for (position, &raw_id) in support.iter().enumerate() {
            let id = raw_id as usize;

            let ix =
                ((x[id] - x0) / scaffold_cell_m).floor() as i64;
            let iy =
                ((y[id] - y0) / scaffold_cell_m).floor() as i64;

            if ix > max_ix {
                max_ix = ix;
            }

            cells.push((ix, position, iy));
        }

        let nx = max_ix + 1;

        let mut keyed: Vec<(i64, usize, i64)> =
            Vec::with_capacity(cells.len());

        for (ix, position, iy) in cells.into_iter() {
            keyed.push((
                iy * nx + ix,
                position,
                support[position],
            ));
        }

        // Stable: preserve #6A / NumPy mergesort behavior.
        keyed.sort_by(|a, b| a.0.cmp(&b.0));

        let mut scaffold: Vec<i64> = Vec::new();
        let mut start = 0usize;

        while start < keyed.len() {
            let key = keyed[start].0;
            let mut end = start + 1;

            while end < keyed.len() && keyed[end].0 == key {
                end += 1;
            }

            // np.argmin first-minimum behavior.
            let mut best = keyed[start].2;
            let mut best_z = z[best as usize];

            for j in (start + 1)..end {
                let candidate = keyed[j].2;
                let candidate_z = z[candidate as usize];

                if candidate_z < best_z {
                    best = candidate;
                    best_z = candidate_z;
                }
            }

            scaffold.push(best);
            start = end;
        }

        if scaffold.len() < min_support_points {
            outputs.push((PyArray1::from_vec(py, Vec::new()), 0));
            continue;
        }

        let two_pi = 2.0_f64 * std::f64::consts::PI;
        let width = two_pi / 8.0_f64;
        let mut seen = [false; 8];

        for &raw_id in scaffold.iter() {
            let id = raw_id as usize;

            let dx = x[id] - px;
            let dy = y[id] - py0;

            let mut angle = dy.atan2(dx) + two_pi;
            angle = angle % two_pi;

            let sector = (angle / width).floor() as usize;

            if sector < 8 {
                seen[sector] = true;
            }
        }

        let sector_count =
            seen.iter().filter(|&&v| v).count();

        if sector_count < min_support_sectors {
            outputs.push((
                PyArray1::from_vec(py, Vec::new()),
                sector_count
            ));
            continue;
        }

        outputs.push((
            PyArray1::from_vec(py, scaffold),
            sector_count
        ));
    }

    Ok(outputs)
}



/*
FASTGC_NATIVE_ULS_SUPPORT_BATCH_V1 — Kernel #7A

Execution-only batching of the accepted #6B4 support preparation.

Scientific invariants:
- neighborhoods originate from the existing scalar SciPy cKDTree query;
- CSR packing preserves each returned neighbor order;
- radius order is unchanged;
- ground filtering, exclusion, lower scaffold, stable cell ordering,
  first-minimum behavior, and sector counting reproduce #6B4.
*/
// FASTGC_KERNEL7B_CACHED_DISTANCE_V1
#[pyfunction]
fn airborne_prepare_support_multi_cached<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    ground: PyReadonlyArray1<'py, bool>,
    nbr_max: PyReadonlyArray1<'py, i64>,
    point_id: i64,
    radii: PyReadonlyArray1<'py, f64>,
    exclusion_radius_m: f64,
    scaffold_cell_m: f64,
    min_support_points: usize,
    min_support_sectors: usize,
) -> PyResult<Vec<(Bound<'py, PyArray1<i64>>, usize)>> {
    let x = x.as_slice()?;
    let y = y.as_slice()?;
    let z = z.as_slice()?;
    let ground = ground.as_slice()?;
    let nbr_max = nbr_max.as_slice()?;
    let radii = radii.as_slice()?;

    let n = x.len();

    if y.len() != n || z.len() != n || ground.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x, y, z, ground length mismatch"
        ));
    }

    if point_id < 0 || point_id as usize >= n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "point_id out of range"
        ));
    }

    let pid = point_id as usize;
    let px = x[pid];
    let py0 = y[pid];

    /*
    Kernel #6B4.

    nbr_max is the ORDERED result of the existing scalar SciPy
    query_ball_point at max(radii).

    Each radius-specific neighborhood is derived by scanning nbr_max
    in that exact order and retaining d <= radius.

    The remainder deliberately reproduces Kernel #6A semantics
    independently for every radius.
    */

    let mut outputs = Vec::with_capacity(radii.len());

    // FASTGC_KERNEL7B_DISTANCE_CACHE_V1
    // Preserve nbr_max order exactly. Compute the same f64::hypot
    // value once per neighbor and reuse it for every configured radius.
    let mut neighbor_distances: Vec<f64> =
        Vec::with_capacity(nbr_max.len());

    for &raw_id in nbr_max.iter() {
        if raw_id < 0 || raw_id as usize >= n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "neighbor id out of range"
            ));
        }

        let id = raw_id as usize;
        let dx = x[id] - px;
        let dy = y[id] - py0;

        neighbor_distances.push(dx.hypot(dy));
    }

    for &radius in radii.iter() {
        if !radius.is_finite() || radius < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "radii must be finite and non-negative"
            ));
        }

        let mut support: Vec<i64> = Vec::new();

        for (&raw_id, &d) in
            nbr_max.iter().zip(neighbor_distances.iter())
        {
            if raw_id < 0 || raw_id as usize >= n {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "neighbor id out of range"
                ));
            }

            let id = raw_id as usize;

            // Exact radius-subset boundary established by #6B3.
            if d > radius {
                continue;
            }

            if !ground[id] || id == pid {
                continue;
            }

            if d >= exclusion_radius_m {
                support.push(raw_id);
            }
        }

        if support.len() < min_support_points {
            outputs.push((PyArray1::from_vec(py, Vec::new()), 0));
            continue;
        }

        let mut x0 = f64::INFINITY;
        let mut y0 = f64::INFINITY;

        for &raw_id in support.iter() {
            let id = raw_id as usize;
            if x[id] < x0 { x0 = x[id]; }
            if y[id] < y0 { y0 = y[id]; }
        }

        let mut cells: Vec<(i64, usize, i64)> =
            Vec::with_capacity(support.len());

        let mut max_ix: i64 = 0;

        for (position, &raw_id) in support.iter().enumerate() {
            let id = raw_id as usize;

            let ix =
                ((x[id] - x0) / scaffold_cell_m).floor() as i64;
            let iy =
                ((y[id] - y0) / scaffold_cell_m).floor() as i64;

            if ix > max_ix {
                max_ix = ix;
            }

            cells.push((ix, position, iy));
        }

        let nx = max_ix + 1;

        let mut keyed: Vec<(i64, usize, i64)> =
            Vec::with_capacity(cells.len());

        for (ix, position, iy) in cells.into_iter() {
            keyed.push((
                iy * nx + ix,
                position,
                support[position],
            ));
        }

        // Stable: preserve #6A / NumPy mergesort behavior.
        keyed.sort_by(|a, b| a.0.cmp(&b.0));

        let mut scaffold: Vec<i64> = Vec::new();
        let mut start = 0usize;

        while start < keyed.len() {
            let key = keyed[start].0;
            let mut end = start + 1;

            while end < keyed.len() && keyed[end].0 == key {
                end += 1;
            }

            // np.argmin first-minimum behavior.
            let mut best = keyed[start].2;
            let mut best_z = z[best as usize];

            for j in (start + 1)..end {
                let candidate = keyed[j].2;
                let candidate_z = z[candidate as usize];

                if candidate_z < best_z {
                    best = candidate;
                    best_z = candidate_z;
                }
            }

            scaffold.push(best);
            start = end;
        }

        if scaffold.len() < min_support_points {
            outputs.push((PyArray1::from_vec(py, Vec::new()), 0));
            continue;
        }

        let two_pi = 2.0_f64 * std::f64::consts::PI;
        let width = two_pi / 8.0_f64;
        let mut seen = [false; 8];

        for &raw_id in scaffold.iter() {
            let id = raw_id as usize;

            let dx = x[id] - px;
            let dy = y[id] - py0;

            let mut angle = dy.atan2(dx) + two_pi;
            angle = angle % two_pi;

            let sector = (angle / width).floor() as usize;

            if sector < 8 {
                seen[sector] = true;
            }
        }

        let sector_count =
            seen.iter().filter(|&&v| v).count();

        if sector_count < min_support_sectors {
            outputs.push((
                PyArray1::from_vec(py, Vec::new()),
                sector_count
            ));
            continue;
        }

        outputs.push((
            PyArray1::from_vec(py, scaffold),
            sector_count
        ));
    }

    Ok(outputs)
}



/*
FASTGC_NATIVE_ULS_SUPPORT_BATCH_V1 — Kernel #7A

Execution-only batching of the accepted #6B4 support preparation.

Scientific invariants:
- neighborhoods originate from the existing scalar SciPy cKDTree query;
- CSR packing preserves each returned neighbor order;
- radius order is unchanged;
- ground filtering, exclusion, lower scaffold, stable cell ordering,
  first-minimum behavior, and sector counting reproduce #6B4.
*/

#[pyfunction]
fn airborne_prepare_support_multi_batch<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    ground: PyReadonlyArray1<'py, bool>,
    candidate_ids: PyReadonlyArray1<'py, i64>,
    neighbor_ids: PyReadonlyArray1<'py, i64>,
    offsets: PyReadonlyArray1<'py, i64>,
    radii: PyReadonlyArray1<'py, f64>,
    exclusion_radius_m: f64,
    scaffold_cell_m: f64,
    min_support_points: usize,
    min_support_sectors: usize,
) -> PyResult<Vec<Vec<(Bound<'py, PyArray1<i64>>, usize)>>> {
    let x = x.as_slice()?;
    let y = y.as_slice()?;
    let z = z.as_slice()?;
    let ground = ground.as_slice()?;
    let candidate_ids = candidate_ids.as_slice()?;
    let neighbor_ids = neighbor_ids.as_slice()?;
    let offsets = offsets.as_slice()?;
    let radii = radii.as_slice()?;

    let n = x.len();

    if y.len() != n || z.len() != n || ground.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x, y, z, ground length mismatch"
        ));
    }

    if offsets.len() != candidate_ids.len() + 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "offsets length must equal candidate_ids length + 1"
        ));
    }

    if offsets.first().copied().unwrap_or(0) != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "offsets must start at zero"
        ));
    }

    if offsets.last().copied().unwrap_or(0) != neighbor_ids.len() as i64 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "final offset must equal neighbor_ids length"
        ));
    }

    for &radius in radii.iter() {
        if !radius.is_finite() || radius < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "radii must be finite and non-negative"
            ));
        }
    }

    let mut batch_outputs = Vec::with_capacity(candidate_ids.len());

    for (candidate_index, &point_id) in candidate_ids.iter().enumerate() {
        if point_id < 0 || point_id as usize >= n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "point_id out of range"
            ));
        }

        let start_raw = offsets[candidate_index];
        let end_raw = offsets[candidate_index + 1];

        if start_raw < 0 || end_raw < start_raw {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "invalid CSR offsets"
            ));
        }

        let start = start_raw as usize;
        let end = end_raw as usize;

        if end > neighbor_ids.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "CSR offset exceeds neighbor_ids length"
            ));
        }

        let nbr_max = &neighbor_ids[start..end];

        let pid = point_id as usize;
        let px = x[pid];
        let py0 = y[pid];

        let mut outputs = Vec::with_capacity(radii.len());

        for &radius in radii.iter() {
            let mut support: Vec<i64> = Vec::new();

            for &raw_id in nbr_max.iter() {
                if raw_id < 0 || raw_id as usize >= n {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "neighbor id out of range"
                    ));
                }

                let id = raw_id as usize;
                let dx = x[id] - px;
                let dy = y[id] - py0;
                let d = dx.hypot(dy);

                if d > radius {
                    continue;
                }

                if !ground[id] || id == pid {
                    continue;
                }

                if d >= exclusion_radius_m {
                    support.push(raw_id);
                }
            }

            if support.len() < min_support_points {
                outputs.push((PyArray1::from_vec(py, Vec::new()), 0));
                continue;
            }

            let mut x0 = f64::INFINITY;
            let mut y0 = f64::INFINITY;

            for &raw_id in support.iter() {
                let id = raw_id as usize;
                if x[id] < x0 { x0 = x[id]; }
                if y[id] < y0 { y0 = y[id]; }
            }

            let mut cells: Vec<(i64, usize, i64)> =
                Vec::with_capacity(support.len());

            let mut max_ix: i64 = 0;

            for (position, &raw_id) in support.iter().enumerate() {
                let id = raw_id as usize;

                let ix =
                    ((x[id] - x0) / scaffold_cell_m).floor() as i64;
                let iy =
                    ((y[id] - y0) / scaffold_cell_m).floor() as i64;

                if ix > max_ix {
                    max_ix = ix;
                }

                cells.push((ix, position, iy));
            }

            let nx = max_ix + 1;

            let mut keyed: Vec<(i64, usize, i64)> =
                Vec::with_capacity(cells.len());

            for (ix, position, iy) in cells.into_iter() {
                keyed.push((
                    iy * nx + ix,
                    position,
                    support[position],
                ));
            }

            // Stable ordering exactly as accepted #6B4.
            keyed.sort_by(|a, b| a.0.cmp(&b.0));

            let mut scaffold: Vec<i64> = Vec::new();
            let mut start_cell = 0usize;

            while start_cell < keyed.len() {
                let key = keyed[start_cell].0;
                let mut end_cell = start_cell + 1;

                while end_cell < keyed.len()
                    && keyed[end_cell].0 == key
                {
                    end_cell += 1;
                }

                // NumPy argmin first-minimum behavior.
                let mut best = keyed[start_cell].2;
                let mut best_z = z[best as usize];

                for j in (start_cell + 1)..end_cell {
                    let candidate = keyed[j].2;
                    let candidate_z = z[candidate as usize];

                    if candidate_z < best_z {
                        best = candidate;
                        best_z = candidate_z;
                    }
                }

                scaffold.push(best);
                start_cell = end_cell;
            }

            if scaffold.len() < min_support_points {
                outputs.push((PyArray1::from_vec(py, Vec::new()), 0));
                continue;
            }

            let two_pi = 2.0_f64 * std::f64::consts::PI;
            let width = two_pi / 8.0_f64;
            let mut seen = [false; 8];

            for &raw_id in scaffold.iter() {
                let id = raw_id as usize;

                let dx = x[id] - px;
                let dy = y[id] - py0;

                let mut angle = dy.atan2(dx) + two_pi;
                angle = angle % two_pi;

                let sector = (angle / width).floor() as usize;

                if sector < 8 {
                    seen[sector] = true;
                }
            }

            let sector_count =
                seen.iter().filter(|&&v| v).count();

            if sector_count < min_support_sectors {
                outputs.push((
                    PyArray1::from_vec(py, Vec::new()),
                    sector_count
                ));
                continue;
            }

            outputs.push((
                PyArray1::from_vec(py, scaffold),
                sector_count
            ));
        }

        batch_outputs.push(outputs);
    }

    Ok(batch_outputs)
}



// FASTGC_KERNEL7C_RAYON_BOUNDED_V1
// Scheduling-only experiment derived mechanically from exact #7A.
// Scientific support/scaffold/sector semantics are unchanged.
#[pyfunction]
fn airborne_prepare_support_multi_batch_rayon<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    ground: PyReadonlyArray1<'py, bool>,
    candidate_ids: PyReadonlyArray1<'py, i64>,
    neighbor_ids: PyReadonlyArray1<'py, i64>,
    offsets: PyReadonlyArray1<'py, i64>,
    radii: PyReadonlyArray1<'py, f64>,
    exclusion_radius_m: f64,
    scaffold_cell_m: f64,
    min_support_points: usize,
    min_support_sectors: usize,
) -> PyResult<Vec<Vec<(Bound<'py, PyArray1<i64>>, usize)>>> {
    let x = x.as_slice()?;
    let y = y.as_slice()?;
    let z = z.as_slice()?;
    let ground = ground.as_slice()?;
    let candidate_ids = candidate_ids.as_slice()?;
    let neighbor_ids = neighbor_ids.as_slice()?;
    let offsets = offsets.as_slice()?;
    let radii = radii.as_slice()?;

    let n = x.len();

    if y.len() != n || z.len() != n || ground.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x, y, z, ground length mismatch"
        ));
    }

    if offsets.len() != candidate_ids.len() + 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "offsets length must equal candidate_ids length + 1"
        ));
    }

    if offsets.first().copied().unwrap_or(0) != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "offsets must start at zero"
        ));
    }

    if offsets.last().copied().unwrap_or(0) != neighbor_ids.len() as i64 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "final offset must equal neighbor_ids length"
        ));
    }

    for &radius in radii.iter() {
        if !radius.is_finite() || radius < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "radii must be finite and non-negative"
            ));
        }
    }

    type RadiusOutput = (Vec<i64>, usize);
    type CandidateOutput = Vec<RadiusOutput>;


    // FASTGC_KERNEL7C_PRE_RAYON_VALIDATION_V2
    //
    // Validate all candidate IDs, CSR boundaries, and neighbor IDs
    // serially before entering the infallible Rayon worker.
    for (candidate_index, &point_id) in
        candidate_ids.iter().enumerate()
    {
        if point_id < 0 || point_id as usize >= n {
            return Err(
                pyo3::exceptions::PyValueError::new_err(
                    "point_id out of range"
                )
            );
        }

        let start_raw = offsets[candidate_index];
        let end_raw = offsets[candidate_index + 1];

        if start_raw < 0 || end_raw < start_raw {
            return Err(
                pyo3::exceptions::PyValueError::new_err(
                    "invalid CSR offsets"
                )
            );
        }

        let start = start_raw as usize;
        let end = end_raw as usize;

        if end > neighbor_ids.len() {
            return Err(
                pyo3::exceptions::PyValueError::new_err(
                    "CSR offset exceeds neighbor_ids length"
                )
            );
        }

        for &raw_id in neighbor_ids[start..end].iter() {
            if raw_id < 0 || raw_id as usize >= n {
                return Err(
                    pyo3::exceptions::PyValueError::new_err(
                        "neighbor id out of range"
                    )
                );
            }
        }
    }

    let raw_outputs: Vec<CandidateOutput> =
        candidate_ids
            .par_iter()
            .enumerate()
            .map(|(candidate_index, &point_id)| {
        let start_raw = offsets[candidate_index];
        let end_raw = offsets[candidate_index + 1];

        let start = start_raw as usize;
        let end = end_raw as usize;

        let nbr_max = &neighbor_ids[start..end];

        let pid = point_id as usize;
        let px = x[pid];
        let py0 = y[pid];

        let mut outputs = Vec::with_capacity(radii.len());

        for &radius in radii.iter() {
            let mut support: Vec<i64> = Vec::new();

            for &raw_id in nbr_max.iter() {
                let id = raw_id as usize;
                let dx = x[id] - px;
                let dy = y[id] - py0;
                let d = dx.hypot(dy);

                if d > radius {
                    continue;
                }

                if !ground[id] || id == pid {
                    continue;
                }

                if d >= exclusion_radius_m {
                    support.push(raw_id);
                }
            }

            if support.len() < min_support_points {
                outputs.push((Vec::new(), 0));
                continue;
            }

            let mut x0 = f64::INFINITY;
            let mut y0 = f64::INFINITY;

            for &raw_id in support.iter() {
                let id = raw_id as usize;
                if x[id] < x0 { x0 = x[id]; }
                if y[id] < y0 { y0 = y[id]; }
            }

            let mut cells: Vec<(i64, usize, i64)> =
                Vec::with_capacity(support.len());

            let mut max_ix: i64 = 0;

            for (position, &raw_id) in support.iter().enumerate() {
                let id = raw_id as usize;

                let ix =
                    ((x[id] - x0) / scaffold_cell_m).floor() as i64;
                let iy =
                    ((y[id] - y0) / scaffold_cell_m).floor() as i64;

                if ix > max_ix {
                    max_ix = ix;
                }

                cells.push((ix, position, iy));
            }

            let nx = max_ix + 1;

            let mut keyed: Vec<(i64, usize, i64)> =
                Vec::with_capacity(cells.len());

            for (ix, position, iy) in cells.into_iter() {
                keyed.push((
                    iy * nx + ix,
                    position,
                    support[position],
                ));
            }

            // Stable ordering exactly as accepted #6B4.
            keyed.sort_by(|a, b| a.0.cmp(&b.0));

            let mut scaffold: Vec<i64> = Vec::new();
            let mut start_cell = 0usize;

            while start_cell < keyed.len() {
                let key = keyed[start_cell].0;
                let mut end_cell = start_cell + 1;

                while end_cell < keyed.len()
                    && keyed[end_cell].0 == key
                {
                    end_cell += 1;
                }

                // NumPy argmin first-minimum behavior.
                let mut best = keyed[start_cell].2;
                let mut best_z = z[best as usize];

                for j in (start_cell + 1)..end_cell {
                    let candidate = keyed[j].2;
                    let candidate_z = z[candidate as usize];

                    if candidate_z < best_z {
                        best = candidate;
                        best_z = candidate_z;
                    }
                }

                scaffold.push(best);
                start_cell = end_cell;
            }

            if scaffold.len() < min_support_points {
                outputs.push((Vec::new(), 0));
                continue;
            }

            let two_pi = 2.0_f64 * std::f64::consts::PI;
            let width = two_pi / 8.0_f64;
            let mut seen = [false; 8];

            for &raw_id in scaffold.iter() {
                let id = raw_id as usize;

                let dx = x[id] - px;
                let dy = y[id] - py0;

                let mut angle = dy.atan2(dx) + two_pi;
                angle = angle % two_pi;

                let sector = (angle / width).floor() as usize;

                if sector < 8 {
                    seen[sector] = true;
                }
            }

            let sector_count =
                seen.iter().filter(|&&v| v).count();

            if sector_count < min_support_sectors {
                outputs.push((
                    Vec::new(),
                    sector_count
                ));
                continue;
            }

            outputs.push((
                scaffold,
                sector_count
            ));
        }

        outputs
            })
            .collect();

    // Rayon is complete here. Convert Rust vectors to NumPy only
    // on the Python thread.
    let mut batch_outputs = Vec::with_capacity(raw_outputs.len());

    for candidate_output in raw_outputs.into_iter() {
        let mut outputs = Vec::with_capacity(candidate_output.len());

        for (ids, sector_count) in candidate_output.into_iter() {
            outputs.push((
                PyArray1::from_vec(py, ids),
                sector_count
            ));
        }

        batch_outputs.push(outputs);
    }

    Ok(batch_outputs)
}

#[pymodule]

fn _fastgc_native(

    m: &Bound<'_, PyModule>

) -> PyResult<()> {



    m.add_function(

        wrap_pyfunction!(

            lower_scaffold,

            m

        )?

    )?;





    m.add_function(

        wrap_pyfunction!(

            robust_stats_mask,

            m

        )?

    )?;





    m.add_function(

        wrap_pyfunction!(

            surface_robust_plane_experimental,

            m

        )?

    )?;



    m.add_function(
        wrap_pyfunction!(
            surface_plane_iteration,
            m
        )?
    )?;
    m.add_function(wrap_pyfunction!(uls_terrain_plane_iteration, m)?)?;
    m.add_function(wrap_pyfunction!(tls_final_plane_iteration, m)?)?;
    m.add_function(
        wrap_pyfunction!(
            airborne_prepare_support,
            m
        )?
    )?;
    m.add_function(
        wrap_pyfunction!(
            airborne_prepare_support_multi,
            m
        )?
    )?;
    m.add_function(
        wrap_pyfunction!(
            airborne_prepare_support_multi_cached,
            m
        )?
    )?;
    m.add_function(
        wrap_pyfunction!(
            airborne_prepare_support_multi_batch,
            m
        )?
    )?;

    m.add_function(
        wrap_pyfunction!(
            airborne_prepare_support_multi_batch_rayon,
            m
        )?
    )?;

    // FASTGC_KERNEL7C_REGISTRATION_V1


    Ok(())

}


