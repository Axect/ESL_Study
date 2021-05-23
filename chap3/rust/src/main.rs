#[macro_use]
extern crate peroxide;
use peroxide::fuga::*;
use itertools::izip;

fn main() {
    // Generate random data
    let x = seq(0, std::f64::consts::PI, 0.01);
    let err = rnorm!(x.len());
    let y = x.zip_with(|t, e| t.sin() + (t / 3f64).powi(2) + 0.1 * e, &err);
    let y_bar = y.mean();

    let mut X = zeros_shape(x.len(), 310, Col);
    for j in 0 .. X.col {
        X.subs_col(j, &x.fmap(|t| phi(j as f64/ 100f64, 0.1f64, t)));
    }

    //let X = hstack!(x.clone(), x.fmap(|t| t.powi(2)), x.fmap(|t| t.powi(3)), x.fmap(|t| t.powi(4)));

    // OLS Estimator
    let mut ols = LinReg::new(&X, &y, Method::OLS);
    ols.estimate();
    ols.stat_test();

    // Ridge
    let mut ridge = LinReg::new(&X, &y, Method::Ridge(1f64));
    ridge.estimate();
    ridge.stat_test();
    println!("RIDGE: ");
    ridge.beta().print();

    println!("");

    // Lasso
    let mut lasso = LinReg::new(&X, &y, Method::Lasso(0.1f64));
    lasso.estimate();
    println!("LASSO: ");
    lasso.beta().print();

    let mut df = DataFrame::new(vec![]);
    df.push("x", Series::new(x));
    df.push("y", Series::new(y));
    df.push("y_ols", Series::new(ols.y_hat().clone()));
    df.push("y_ridge", Series::new(ridge.y_hat().add_s(y_bar)));
    df.push("y_lasso", Series::new(lasso.y_hat().clone()));

    df.print();

    df.write_nc("data/lasso.nc").expect("Can't write nc file");
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
    let svd = x.svd().truncated();
    let u = svd.u();
    ((u * &u.t()).apply(&y), beta)
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
// Gaussian Basis
// =============================================================================
pub fn phi(j: f64, s: f64, x: f64) -> f64 {
    let mu = j;
    (-(x - mu).powi(2) / s).exp()
}

// =============================================================================
// Coordinate Descent
// =============================================================================
#[allow(non_snake_case)]
fn coordinate_descent_lasso(beta_init: &Vec<f64>, X: &Matrix, y: &Vec<f64>, lam: f64, num_iters: usize) -> Vec<f64> {
    let N = X.row;
    let p = X.col;
    let mut beta = beta_init.clone();

    for i in 0 .. num_iters {
        for j in 0 .. p {
            let y_hat = X * &beta;
            let x_j = X.col(j);
            let rho = x_j.dot(
                &izip!(y, &y_hat, &x_j)
                    .map(|(a, b, c)| a - b + c * beta[j])
                    .collect::<Vec<f64>>()
            );
            beta[j] = soft_threshold(rho, lam);
        }
    }
    beta
}

fn soft_threshold(rho: f64, lam: f64) -> f64 {
    if rho < -lam {
        rho + lam
    } else if rho > lam {
        rho - lam
    } else {
        0f64
    }
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
    beta_0: Option<f64>,
    y_hat: Option<Vec<f64>>,
    sigma_hat: Option<f64>,
    t_score: Option<Vec<f64>>,
    p_value: Option<Vec<f64>>,
    method: Method,
    _cached_svd: Option<SVD>,
}

#[derive(Debug, Clone, Copy)]
pub enum Method {
    OLS,
    Ridge(f64),
    Lasso(f64),
}

impl LinReg {
    pub fn new(x: &Matrix, y: &Vec<f64>, method: Method) -> Self {
        let (x_mat, y_vec, beta, beta_0) = match method {
            Method::OLS => {
                (add_bias(x.clone()), y.clone(), None, None)
            }
            Method::Ridge(_) => {
                (x.standardize(), y.centered(), None, Some(y.mean()))
            }
            Method::Lasso(_) => {
                (x.centered().normalized(), y.centered(), Some(vec![1f64; x.col]), Some(y.mean()))
            }
        };
        
        Self {
            input: x_mat,
            output: y_vec,
            N: y.len(),
            p: x.col,
            beta,
            beta_0,
            y_hat: None,
            sigma_hat: None,
            t_score: None,
            p_value: None,
            method,
            _cached_svd: None,
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

    pub fn beta_0(&self) -> &f64 {
        match &self.beta_0 {
            None => panic!("Not inserted!"),
            Some(beta_0) => beta_0,
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
    
    fn cached_svd(&self) -> &SVD {
        match &self._cached_svd {
            None => panic!("SVD is not yet calculated"),
            Some(svd) => svd
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
                // Truncated SVD
                let x_svd = self.input.svd().truncated();
                let s_star: Vec<f64> = x_svd.s.iter()
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

                let y_hat = &(add_bias(self.input().clone())) * &beta;

                self.beta = Some(beta);
                self.y_hat = Some(y_hat);
                self._cached_svd = Some(x_svd);
            }
            Method::Lasso(lam) => {
                let beta_hat = coordinate_descent_lasso(self.beta(), self.input(), self.output(), lam, 200);
                let y_hat = (self.input() * &beta_hat).add_s(*self.beta_0());

                self.beta = Some(beta_hat);
                self.y_hat = Some(y_hat);
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
                let x_svd = self.cached_svd();
                let u = x_svd.u();
                let s = &x_svd.s;
                let s_mat = x_svd.s_mat();
                let vt = x_svd.vt();
                let v = vt.t();
                let s_star = s.fmap(|t| t / (t.powi(2) + lam));
                let mut sigma_star = zeros(s.len(), s.len());
                for i in 0 .. s.len() {
                    sigma_star[(i, i)] = s_star[i].powi(2);
                }
                let v_mat = &(&v * &sigma_star) * vt;
                let v_vec = v_mat.diag();

                let nu = (N - p) as f64 + s.fmap(|t| lam / (t.powi(2) + lam)).sum();
                let rss = calc_rss(self.output(), self.y_hat());
                let sigma_hat = rss / nu;
                let beta = self.beta().skip(1);
                let t_score = beta.zip_with(|b_j, v_j| b_j / (sigma_hat * v_j).sqrt(), &v_vec);
                let t_dist = OPDist::StudentT(nu);
                let p_value = t_score.fmap(|t| calc_p_value(&t_dist, t));

                self.sigma_hat = Some(sigma_hat);
                self.t_score = Some(t_score);
                self.p_value = Some(p_value);
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
    fn normalized(&self) -> Self;
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

    fn normalized(&self) -> Self {
        let norm = self.norm(Norm::L2);
        self.div_s(norm)
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

    fn normalized(&self) -> Self {
        self.col_map(|c| c.div_s(c.norm(Norm::L2)))
    }
}
