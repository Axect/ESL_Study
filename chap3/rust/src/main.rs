#[macro_use]
extern crate peroxide;
use peroxide::fuga::*;

fn main() {
    // Generate random data
    let x = seq(1, 5, 0.1);
    let err = rnorm!(x.len());
    let y = x.zip_with(|x, e| 2f64 * x + 3f64 + e, &err);

    // OLS Estimator
    let mut ols = OLSEstimator::new(&x, &y);
    ols.estimate();
    ols.stat_test();

    let y_hat= ols.y_hat();
    let sigma_hat = ols.sigma_hat();
    let t_score = ols.t_score();
    let p_value = ols.p_value();

    sigma_hat.print();
    t_score.print();
    println!("{:?}", p_value);

    // Save data to plot
    let mut df = DataFrame::new(vec![]);
    df.push("x", Series::new(x));
    df.push("y", Series::new(y));
    df.push("y_hat", Series::new(y_hat.clone()));

    df.print();
    //df.write_nc("data/data.nc").expect("Can't write nc file");
}

// =============================================================================
// Estimate Procedure
// =============================================================================
fn add_bias(x: Matrix) -> Matrix {
    let one = vec![1f64; x.row];
    cbind(one.into(), x)
}

fn find_beta_hat(x: &Matrix, y: &Vec<f64>) -> Vec<f64> {
    &x.pseudo_inv() * y
}

fn find_y_hat(x: &Matrix, y: &Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    let beta = find_beta_hat(x, y);
    (x * &beta, beta)
}

// =============================================================================
// Test Procedure
// =============================================================================
fn find_sigma_hat(y: &Vec<f64>, y_hat: &Vec<f64>, p: usize) -> f64 {
    let mut s = 0f64;
    for (a, b) in y.iter().zip(y_hat.iter()) {
        s += (a - b).powi(2);
    }
    s / (y.len() - p - 1) as f64
}

fn calc_t_score(beta: &Vec<f64>, x: &Matrix, sigma: f64) -> Vec<f64> {
    let v = (&x.t() * x).inv().diag();
    beta.zip_with(|b, v| b / (v * sigma).sqrt(), &v)
}

fn calc_p_value<D: RNG>(dist: &D, z: f64) -> f64 {
    (1f64 - dist.cdf(z)) * 2f64
}

// =============================================================================
// OOP implementation
// =============================================================================
#[derive(Debug, Clone)]
pub struct OLSEstimator {
    input: Matrix,
    output: Vec<f64>,
    beta: Option<Vec<f64>>,
    y_hat: Option<Vec<f64>>,
    sigma_hat: Option<f64>,
    t_score: Option<Vec<f64>>,
    p_value: Option<Vec<f64>>,
}

impl OLSEstimator {
    pub fn new(x: &Vec<f64>, y: &Vec<f64>) -> Self {
        let x_mat = add_bias(x.into());
        
        Self {
            input: x_mat,
            output: y.clone(),
            beta: None,
            y_hat: None,
            sigma_hat: None,
            t_score: None,
            p_value: None,
        }
    } 

    pub fn input(&self) -> &Matrix {
        &self.input
    }

    pub fn output(&self) -> &Vec<f64> {
        &self.output
    }

    pub fn beta(&self) -> &Vec<f64> {
        match &self.beta {
            None => panic!("Not yet estimated!"),
            Some(beta) => beta,
        }
    }

    pub fn y_hat(&self) -> &Vec<f64> {
        match &self.y_hat {
            None => panic!("Not yet estimated!"),
            Some(y_hat) => y_hat
        }
    }

    pub fn sigma_hat(&self) -> f64 {
        match self.sigma_hat {
            None => panic!("Not yet tested!"),
            Some(sigma_hat) => sigma_hat
        }
    }

    pub fn t_score(&self) -> &Vec<f64> {
        match &self.t_score {
            None => panic!("Not yet tested!"),
            Some(t_score) => t_score
        }
    }

    pub fn p_value(&self) -> &Vec<f64> {
        match &self.p_value {
            None => panic!("Not yet tested!"),
            Some(p_value) => p_value
        }
    }

    pub fn estimate(&mut self) {
        let (y_hat, beta) = find_y_hat(self.input(), self.output());
        self.beta = Some(beta);
        self.y_hat = Some(y_hat);
    }

    #[allow(non_snake_case)]
    pub fn stat_test(&mut self) {
        let N: usize = self.output().len();
        let p: usize = self.input().col-1;

        let sigma_hat = find_sigma_hat(self.output(), self.y_hat(), p);
        let t_score = calc_t_score(self.beta(), self.input(), sigma_hat);

        let t_dist = OPDist::StudentT((N-p-1) as f64);

        let p_value = t_score.fmap(|t| calc_p_value(&t_dist, t));

        self.sigma_hat = Some(sigma_hat);
        self.t_score = Some(t_score);
        self.p_value = Some(p_value);
    }
}