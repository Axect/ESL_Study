use peroxide::fuga::*;
use rayon::prelude::*;
use std::f64::consts::PI;
use crate::KernelType::*;

const N: usize = 400;

fn main() {
    let u = Uniform(0f64, 6f64);
    let n = Normal(0f64, 1f64);
    let x = u.sample(N);
    let eps = n.sample(N);
    let y = zip_with(|t, e| 2f64 * t.sin() + t + e, &x, &eps);

    let data = Data {
        x,
        y
    };

    let kernel_config = [Gaussian, Epanechnikov, Tricube]
        .into_iter()
        .map(|k| {
            KernelConfig {
                kernel_type: k,
                lambda: 0.5
            }
        })
        .collect::<Vec<_>>();

    let domain = linspace(0f64, 6f64, N);
    let y = domain.fmap(|t| 2f64 * t.sin() + t);
    let nadaraya_watson_result = kernel_config.iter().map(|k| {
        nadaraya_watson(&domain, *k, &data)
    }).collect::<Vec<_>>();

    let mut data_sorted = data.x.iter().zip(data.y.iter()).collect::<Vec<_>>();
    data_sorted.sort_by(|a, b| a.0.partial_cmp(b.0).unwrap());

    let (x_data, y_data) = data_sorted.into_iter().unzip();
    let data_sorted = Data {
        x: x_data,
        y: y_data
    };

    let attentions = kernel_config.iter().map(|k| {
        attention(&domain, *k, &data_sorted)
    }).collect::<Vec<_>>();

    let mut df = DataFrame::new(vec![]);
    df.push("x_data", Series::new(data.x.clone()));
    df.push("y_data", Series::new(data.y.clone()));
    df.push("x", Series::new(domain.clone()));
    df.push("y", Series::new(y));
    df.push("gaussian", Series::new(nadaraya_watson_result[0].clone()));
    df.push("epanechnikov", Series::new(nadaraya_watson_result[1].clone()));
    df.push("tricube", Series::new(nadaraya_watson_result[2].clone()));

    df.print();

    df.write_parquet("nadaraya_watson.parquet", CompressionOptions::Uncompressed).unwrap();

    let mut dg = DataFrame::new(vec![]);
    dg.push("att_gaussian", Series::new(attentions[0].data.clone()));
    dg.push("att_epanechnikov", Series::new(attentions[1].data.clone()));
    dg.push("att_tricube", Series::new(attentions[2].data.clone()));
    dg.push("row", Series::new(vec![domain.len() as u64]));
    dg.push("col", Series::new(vec![data.x.len() as u64]));
    dg.print();

    dg.write_parquet("attention.parquet", CompressionOptions::Uncompressed).unwrap();
}

#[derive(Debug, Clone)]
struct Data {
    x: Vec<f64>,
    y: Vec<f64>,
}

fn nadaraya_watson(domain: &[f64], kernel_config: KernelConfig, data: &Data) -> Vec<f64> {
    let kernel_sum: Vec<f64> = domain.par_iter().map(|x| {
        data.x.iter().fold(0.0, |acc, x_i| {
            acc + kernel(*x, *x_i, kernel_config)
        })
    }).collect();
    domain.par_iter().zip(kernel_sum.par_iter()).map(|(x, s)| {
        data.x.iter().zip(data.y.iter()).fold(0.0, |acc, (x_i, y_i)| {
            acc + y_i * kernel(*x, *x_i, kernel_config)
        }) / s
    }).collect()
}

fn attention(domain: &[f64], kernel_config: KernelConfig, data: &Data) -> Matrix {
    let kernel_sum = domain.iter().fold(0.0, |acc, x_i| {
        acc + data.x.iter().fold(0.0, |acc, x_j| {
            acc + kernel(*x_i, *x_j, kernel_config)
        })
    });
    let kernel_matrix = domain.par_iter().map(|x_i| {
        data.x.par_iter().map(|x_j| {
            kernel(*x_i, *x_j, kernel_config)
        }).collect::<Vec<_>>()
    }).collect::<Vec<_>>();

    py_matrix(kernel_matrix)
}

fn kernel(x: f64, x_i: f64, kernel_config: KernelConfig) -> f64 {
    let KernelConfig {
        kernel_type,
        lambda
    } = kernel_config;
    match kernel_type {
        KernelType::Gaussian => gaussian((x - x_i) / lambda),
        KernelType::Epanechnikov => epanechnikov((x - x_i) / lambda),
        KernelType::Tricube => tricube((x - x_i) / lambda),
    }
}

#[derive(Debug, Copy, Clone)]
struct KernelConfig {
    kernel_type: KernelType,
    lambda: f64,
}

#[derive(Debug, Copy, Clone)]
enum KernelType {
    Gaussian,
    Epanechnikov,
    Tricube
}

fn gaussian(x: f64) -> f64 {
    1.0 / (2.0 * PI).sqrt() * (-x.powi(2) / 2f64).exp()
}

fn epanechnikov(x: f64) -> f64 {
    if x.abs() > 1.0 {
        0.0
    } else {
        3.0 / 4.0 * (1.0 - x.powi(2))
    }
}

fn tricube(x: f64) -> f64 {
    if x.abs() > 1.0 {
        0.0
    } else {
        70.0 / 81.0 * (1.0 - x.abs().powi(3)).powi(3)
    }
}
