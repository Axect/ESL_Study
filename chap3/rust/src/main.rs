#[macro_use]
extern crate peroxide;
use peroxide::fuga::*;

fn main() {
    // Generate random data
    let x = seq(1, 5, 0.1);
    let err = rnorm!(x.len());
    let y = x.zip_with(|x, e| 2f64 * x + 3f64 + e, &err);

    // Add Bias to X
    let x_mat: Matrix = add_bias(x.clone().into());

    // Find y_hat
    let y_hat = find_y_hat(&x_mat, &y);

    let mut df = DataFrame::new(vec![]);
    df.push("x", Series::new(x));
    df.push("y", Series::new(y));
    df.push("y_hat", Series::new(y_hat));

    df.print();
    df.write_nc("data/data.nc").expect("Can't write nc file");
}

fn add_bias(x: Matrix) -> Matrix {
    let one = vec![1f64; x.row];
    cbind(one.into(), x)
}

fn find_beta_hat(x: &Matrix, y: &Vec<f64>) -> Vec<f64> {
    &x.pseudo_inv() * y
}

fn find_y_hat(x: &Matrix, y: &Vec<f64>) -> Vec<f64> {
    let beta = find_beta_hat(x, y);
    x * &beta
}
