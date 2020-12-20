use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use structopt::StructOpt;

use proc_macro2::TokenStream;
use quote::{ToTokens, TokenStreamExt};
use serde_derive::{Deserialize, Serialize};
use std::str::FromStr;

#[macro_use]
extern crate structopt;

#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate quote;

#[macro_use]
extern crate cmd_lib;

#[derive(StructOpt)]
pub struct GenerationParameters {
    pub output_header: String,
    #[structopt(required = true)]
    pub input_networks: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct LayerDescription {
    size: usize,
    activation: String,
    weights: Vec<Vec<f64>>,
    bias: Vec<f64>,
}

#[derive(Serialize, Deserialize, Debug)]
struct NetworkDescription {
    layers: Vec<LayerDescription>,
}

fn raw_token<T: ToString>(value: T) -> TokenStream {
    TokenStream::from_str(&value.to_string()).unwrap()
}

pub fn generate_headers(params: GenerationParameters) {
    let netdescs: Vec<(String, NetworkDescription)> = params
        .input_networks
        .iter()
        .map(|input_network| {
            let name = Path::new(input_network)
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string();

            (
                name,
                serde_json::from_reader(File::open(input_network).unwrap()).unwrap(),
            )
        })
        .collect();
    let outfile = params.output_header;

    // Workaround for escaping
    let define = TokenStream::from_str("#define").unwrap();
    let include = TokenStream::from_str("#include").unwrap();
    let ifndef = TokenStream::from_str("#ifndef").unwrap();
    let newline = TokenStream::from_str("NEWLINE_TOKEN").unwrap();

    let max_size_layer = raw_token(
        netdescs
            .iter()
            .map(|netdesc| netdesc.1.layers.iter().map(|l| l.size).max().unwrap())
            .max()
            .unwrap(),
    );

    let mut ccode = TokenStream::new();

    ccode.append_all(quote! {

        #ifndef FANN_FANN_NET_H_ #newline
        #define FANN_FANN_NET_H_ #newline #newline

        #include "fann.h" #newline
        #include "fann_structs.h" #newline #newline

        #define TEMP_BUFFER_SIZE #max_size_layer #newline
    });

    for (network_name, network_desc) in netdescs {
        let layers_count = raw_token(network_desc.layers.len());

        for (index, layer) in network_desc.layers.iter().enumerate() {
            let mut flat_weights: Vec<f64> = Vec::new();

            for neuron in &layer.weights {
                flat_weights.append(&mut neuron.clone())
            }

            let warray = raw_token(
                flat_weights
                    .iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );

            let barray = raw_token(
                layer
                    .bias
                    .iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );

            let weights = raw_token(format!("{}_layer_weights_{}", network_name, index));
            let bias = raw_token(format!("{}_layer_bias_{}", network_name, index));

            ccode.append_all(quote! {
                fann_type #weights[] = {#warray}; #newline #newline
                fann_type #bias[] = {#barray}; #newline #newline
            });
        }

        {
            let mut layers_code = TokenStream::new();

            for (index, layer) in network_desc.layers.iter().enumerate() {
                if index > 0 {
                    layers_code.append_all(raw_token(","));
                    layers_code.append_all(newline.clone());
                    layers_code.append_all(newline.clone());
                }

                let weights = raw_token(format!("{}_layer_weights_{}", network_name, index));
                let bias = raw_token(format!("{}_layer_bias_{}", network_name, index));

                let input_count = raw_token(layer.weights[0].len());
                let neurons_count = raw_token(layer.weights.len());

                let activation_function =
                    raw_token(format!("fann_activation_{}", layer.activation));

                layers_code.append_all(quote! {
                    { #newline #newline
                        .inputs_count = #input_count,
                        .neurons_count = #neurons_count,
                        .weights = #weights,
                        .bias = #bias,
                        .activation_function = #activation_function #newline #newline
                    }
                });
            }

            let network_varname = raw_token(format!("{}_{}", network_name, "fann_network"));

            ccode.append_all(quote! {
                const fann_network #network_varname = {
                    .layers_count = #layers_count,
                    .layers = { #newline #layers_code #newline #newline } #newline #newline
                };
            });
        }
    }

    ccode.append_all(raw_token("#endif"));

    let mut ccode_str = ccode.to_string().replace("NEWLINE_TOKEN", "\n");
    ccode_str.push_str(" // FANN_FANN_NET_H_\n");

    File::create(&outfile)
        .unwrap()
        .write_all(ccode_str.as_bytes())
        .unwrap();

    run_cmd!(clang-format -i $outfile).unwrap();
}

pub fn generate_json(pickle_file: impl AsRef<Path>, out_file: impl AsRef<Path>) {
    let generator = String::from(env!("CARGO_MANIFEST_DIR")) + "/generator.py";
    let pickle_file = pickle_file.as_ref().to_string_lossy();
    let out_file = out_file.as_ref().to_string_lossy();
    run_cmd!(python3 $generator $pickle_file $out_file).unwrap();
}
