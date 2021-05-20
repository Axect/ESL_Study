#[macro_use]
extern crate peroxide;
use peroxide::fuga::*;

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

    ols.summary();
    println!("");

    ridge.summary();

    // Lasso
    let mut X_s = X;
    X_s.col_mut_map(|c| c.div_s(c.norm(Norm::L2)));
    let y_c = y.centered();

    let beta_init = find_beta_hat(&X_s, &y);

    let beta = lasso(&beta_init, &X_s, &y, 1e-10, 500);
    beta.print();
    let y_new = (X_s * beta);

    let mut df = DataFrame::new(vec![]);
    df.push("x", Series::new(x));
    df.push("y", Series::new(y));
    df.push("y_ols", Series::new(ols.y_hat().clone()));
    df.push("y_ridge", Series::new(ridge.y_hat().add_s(y_bar)));
    df.push("y_lasso", Series::new(y_new));

    df.print();

    df.write_nc("data/lasso.nc").expect("Can't write nc file");


    //// OLS Estimator
    //let mut ols = LinReg::new(&x.clone().into(), &y, Method::OLS);
    //ols.estimate();
    //ols.stat_test();

    //let mut ridge = LinReg::new(&x.clone().into(), &y, Method::Ridge(1f64));
    //ridge.estimate();
    //ridge.stat_test();

    //ols.summary();
    //println!("");

    //ridge.summary();

    //let beta_hat = ols.beta();
    //let y_hat= ols.y_hat();
    //let sigma_hat = ols.sigma_hat();
    //let t_score = ols.t_score();
    //let p_value = ols.p_value();

    //// Save data to plot
    //let mut df = DataFrame::new(vec![]);
    //df.push("x", Series::new(x));
    //df.push("y", Series::new(y));
    //df.push("y_hat", Series::new(y_hat.clone()));

    //df.print();
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
fn lasso(beta_init: &Vec<f64>, X: &Matrix, y: &Vec<f64>, lam: f64, num_iters: usize) -> Vec<f64> {
    let N = X.row;
    let p = X.col;
    let mut beta = beta_init.clone();

    for i in 0 .. num_iters {
        for j in 0 .. p {
            let y_hat = X * &beta;
            let x_j = X.col(j);
            let rho = x_j.dot(&(y.sub_v(&y_hat).add_v(&x_j.mul_s(beta[j]))));
            beta[j] = soft_threshold(rho, lam);
        }
    }
    beta
}


fn soft_threshold(beta: f64, lam: f64) -> f64 {
    if beta < -lam {
        beta + lam
    } else if beta > lam {
        beta - lam
    } else {
        0f64
    }
}

// =============================================================================
// LAR
// =============================================================================
///// # Condition
///// * X should be standardized
///// * y should be centered
//#[allow(non_snake_case)]
//fn lar(y: &Vec<f64>, X: &Matrix, alpha_origin: f64) -> Vec<f64> {
//    // Initial Step (k=1)
//    let mut alpha = alpha_origin;
//    let mut y_hat = vec![0f64; y.len()];
//    let mut r = y.clone();
//    let mut beta = vec![0f64; X.col];
//    let mut A = vec![];
//    let mut j = find_max_corr(X, &r, &A).unwrap();
//    j.print();
//    A.push(j);
//    let mut j_prev = A.iter().last().unwrap().clone();
//    let mut X_A: Matrix = X.col(j).into();
//    let mut G_A = X_A.pseudo_inv();
//    let mut delta = &G_A * &r;
//    let mut beta_temp = delta.mul_s(alpha);
//    A.iter().zip(beta_temp.iter()).for_each(|(i, b)| beta[*i] = *b);
//    A.print();
//    beta_temp.print();
//    
//    'outer: for k in 0 .. X.col {
//        r = r.sub_v(&X_A.apply(&beta));
//        match find_max_corr(X, &r, &A) {
//            Some(i) => j = i,
//            None => break,
//        }
//        //let mut stack = 0usize;
//        while j == j_prev {
//            let delta_prev = delta.clone();
//            delta = &G_A * &r;
//            if delta[0] * delta_prev[0] < 0f64 {
//                alpha *= 0.01;
//                delta = delta_prev;
//            }
//            beta_temp = beta_temp.add_v(&delta.mul_s(alpha));
//            r = r.sub_v(&X_A.apply(&beta_temp));
//            match find_max_corr(X, &r, &A) {
//                Some(i) => j = i,
//                None => break 'outer,
//            }
//            delta.print();
//            //stack += 1;
//            //stack.print();
//        }
//        alpha = alpha_origin;
//        j_prev = j;
//        A.print();
//        A.iter().zip(beta_temp.iter()).for_each(|(i, b)| beta[*i] = *b);
//        if k == X.col-1 {
//            break
//        }
//        j.print();
//        A.push(j);
//        X_A.add_col_mut(&X.col(j));
//        G_A = X_A.pseudo_inv();
//        delta = G_A.apply(&r);
//        beta_temp.push(0f64);
//        beta_temp = beta_temp.add_v(&delta.mul_s(alpha));
//    }
//    A.iter().zip(beta_temp.iter()).for_each(|(i, b)| beta[*i] = *b);
//    beta
//}
//
//fn find_max_corr(X: &Matrix, target: &Vec<f64>, ignore: &Vec<usize>) -> Option<usize> {
//    let ref_vec = X.col_reduce(|c| cor(&c, target));
//    let mut removed = vec![-2f64; ref_vec.len()];
//    let mut stack = 0usize;
//    for (i, v) in ref_vec.into_iter().enumerate() {
//        if !ignore.contains(&i) {
//            removed[i] = v;
//            stack += 1;
//        }
//    }
//    match stack {
//        0 => None,
//        _ => Some(removed.arg_max())
//    }
//}

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

                println!("beta: {:?}", beta);
                let y_hat = &(add_bias(self.input().clone())) * &beta;

                self.beta = Some(beta);
                self.y_hat = Some(y_hat);
                self._cached_svd = Some(x_svd);
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
