#[macro_use]
extern crate peroxide;
use peroxide::fuga::*;

fn main() {
    // Generate random data
    let x = seq(1, 5, 0.1);
    let err = rnorm!(x.len());
    let y = x.zip_with(|x, e| 2f64 * x + 3f64 + e, &err);
    let y_bar = y.mean();

    // OLS Estimator
    let mut ols = LinReg::new(&x.clone().into(), &y, Method::OLS);
    ols.estimate();
    ols.stat_test();

    let mut ridge = LinReg::new(&x.clone().into(), &y, Method::Ridge(1f64));
    ridge.estimate();
    ridge.stat_test();

    ols.summary();
    println!("");

    ridge.summary();

    let beta_hat = ols.beta();
    let y_hat= ols.y_hat();
    let sigma_hat = ols.sigma_hat();
    let t_score = ols.t_score();
    let p_value = ols.p_value();

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
fn calc_rss(y: &Vec<f64>, y_hat: &Vec<f64>) -> f64 {
    y.zip_with(|x, xh| (x - xh).powi(2), y_hat).sum()
}

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
#[allow(non_snake_case)]
#[derive(Debug, Clone)]
pub struct LinReg {
    input: Matrix,
    output: Vec<f64>,
    N: usize,
    p: usize,
    beta: Option<Vec<f64>>,
    y_hat: Option<Vec<f64>>,
    sigma_hat: Option<f64>,
    t_score: Option<Vec<f64>>,
    p_value: Option<Vec<f64>>,
    method: Method
}

#[derive(Debug, Clone, Copy)]
pub enum Method {
    OLS,
    Ridge(f64),
    Lasso(f64),
}

impl LinReg {
    pub fn new(x: &Matrix, y: &Vec<f64>, method: Method) -> Self {
        let (x_mat, y_vec) = match method {
            Method::OLS => {
                (add_bias(x.clone()), y.clone())
            }
            _ => {
                (x.standardize(), y.centered())
            }
        };
        
        Self {
            input: x_mat,
            output: y_vec,
            N: y.len(),
            p: x.col,
            beta: None,
            y_hat: None,
            sigma_hat: None,
            t_score: None,
            p_value: None,
            method,
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
        match self.method {
            Method::OLS => {
                let (y_hat, beta) = find_y_hat(self.input(), self.output());
                self.beta = Some(beta);
                self.y_hat = Some(y_hat);
            }
            Method::Ridge(lam) => {
                // Full SVD
                let x_svd = self.input.svd();
                let s_star: Vec<f64> = x_svd.s.iter()
                    .filter(|&t| *t != 0f64)
                    .map(|sigma| sigma / (sigma.powi(2) + lam))
                    .collect();
                let u = x_svd.u();
                let vt = x_svd.vt();
                let u = u.take_col(s_star.len());
                let vt = vt.take_row(s_star.len());
                
                let mut sigma_star = zeros(s_star.len(), s_star.len());
                for i in 0 .. s_star.len() {
                    sigma_star[(i, i)] = s_star[i];
                }

                let beta_ridge = vt.t() * sigma_star * (&u.t() * self.output());
                let beta_0 = self.output().mean();

                let beta = cat(beta_0, &beta_ridge);

                println!("beta: {:?}", beta);
                let y_hat = &(add_bias(self.input().clone())) * &beta;

                self.beta = Some(beta);
                self.y_hat = Some(y_hat);
            }
            Method::Lasso(_lam) => {
                todo!()
            }
        }
    }

    #[allow(non_snake_case)]
    pub fn stat_test(&mut self) {
        let N: usize = self.N;
        let p: usize = self.p;

        match self.method {
            Method::OLS => {
                let nu = (N - p - 1) as f64;
                let sigma_hat = find_sigma_hat(self.output(), self.y_hat(), p);
                let t_score = calc_t_score(self.beta(), self.input(), sigma_hat);
                let t_dist = OPDist::StudentT(nu);
                let p_value = t_score.fmap(|t| calc_p_value(&t_dist, t));
                self.sigma_hat = Some(sigma_hat);
                self.t_score = Some(t_score);
                self.p_value = Some(p_value);
            }
            Method::Ridge(lam) => {
                let x_svd = self.input().svd();
                let u = x_svd.u();
                let s = &x_svd.s;
                let vt = x_svd.vt();

                todo!()

            }
            Method::Lasso(lam) => {
                todo!()
            }
        }

    }

    pub fn summary(&self) {
        println!("N: {}", self.N);
        println!("p: {}", self.p);
        print!("beta: ");
        self.beta().print();
        print!("sigma: ");
        self.sigma_hat().print();
        print!("t-score: ");
        self.t_score().print();
        print!("p-value: ");
        self.p_value().print();
    }
}

pub trait Scaled {
    fn standardize(&self) -> Self;
    fn centered(&self) -> Self;
}

impl Scaled for Vec<f64> {
    fn standardize(&self) -> Self {
        let x_bar = self.mean();
        let sd = self.sd();
        self.fmap(|t| (t - x_bar) / sd)
    }

    fn centered(&self) -> Self {
        let y_bar = self.mean();
        self.sub_s(y_bar)
    }
}

impl Scaled for Matrix {
    fn standardize(&self) -> Self {
        let x_bar = self.mean();
        let sd = self.sd();

        let mut m = zeros(self.row, self.col);
        for i in 0 .. self.col {
            let v = self.col(i).fmap(|x| (x - x_bar[i]) / sd[i]);
            m.subs_col(i, &v);
        }
        m
    }

    fn centered(&self) -> Self {
        let x_bar = self.mean();

        let mut m = zeros(self.row, self.col);
        for i in 0 .. self.col {
            let v = self.col(i).sub_s(x_bar[i]);
            m.subs_col(i, &v);
        }
        m
    }
}
