use std::path::Path;
use structopt::StructOpt;
use std::fs::File;
use std::process::Command;
use std::io::Write;

use serde_derive::{Serialize, Deserialize};
use proc_macro2::TokenStream;
use quote::{TokenStreamExt, ToTokens};
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
struct Args {
    input_network: String,
    output_header: String,
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
    layers: Vec<LayerDescription>
}

fn raw_token<T: ToString>(value: T) -> TokenStream {
    TokenStream::from_str(&value.to_string()).unwrap()
}

fn main() {
    let args: Args = Args::from_args();

    let netdesc: NetworkDescription = serde_json::from_reader(File::open(args.input_network).unwrap()).unwrap();
    let outfile = args.output_header;

    // Workaround for escaping
    let define = TokenStream::from_str("#define").unwrap();
    let include = TokenStream::from_str("#include").unwrap();
    let ifndef = TokenStream::from_str("#ifndef").unwrap();
    let newline = TokenStream::from_str("NEWLINE_TOKEN").unwrap();

    let max_size_layer = raw_token(netdesc.layers.iter().map(|l| l.size).max().unwrap());
    let layers_count = raw_token(netdesc.layers.len());


    let mut ccode = TokenStream::new();

    ccode.append_all(quote! {

        #ifndef FANN_FANN_NET_H_ #newline
        #define FANN_FANN_NET_H_ #newline #newline

        #include "fann.h" #newline
        #include "fann_structs.h" #newline #newline

        #define TEMP_BUFFER_SIZE #max_size_layer #newline
        #define NUM_LAYERS #layers_count #newline #newline
    });

    for (index, layer) in netdesc.layers.iter().enumerate() {
        let mut flat_weights: Vec<f64> = Vec::new();

        for neuron in &layer.weights {
            flat_weights.append(&mut neuron.clone())
        }

        let warray = raw_token(flat_weights.iter().map(|f| f.to_string())
            .collect::<Vec<_>>().join(","));

        let barray = raw_token(layer.bias.iter().map(|f| f.to_string())
            .collect::<Vec<_>>().join(","));

        let weights = raw_token(format!("layer_weights_{}", index));
        let bias = raw_token(format!("layer_bias_{}", index));

        ccode.append_all(quote! {
            fann_type #weights[] = {#warray}; #newline #newline
            fann_type #bias[] = {#barray}; #newline #newline
        });
    }

    {
        let mut layers_code = TokenStream::new();

        for (index, layer) in netdesc.layers.iter().enumerate() {

            if index > 0 {
                layers_code.append_all(raw_token(","));
                layers_code.append_all(newline.clone());
                layers_code.append_all(newline.clone());
            }

            let weights = raw_token(format!("layer_weights_{}", index));
            let bias = raw_token(format!("layer_bias_{}", index));

            let input_count = raw_token(layer.weights[0].len());
            let neurons_count = raw_token(layer.weights.len());

            let activation_function = raw_token(format!("fann_activation_{}", layer.activation));

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

        ccode.append_all(quote! {
            const fann_layer fann_layers[] = { #newline #layers_code #newline #newline}; #newline #newline
        });
    }

    ccode.append_all(raw_token("#endif"));

    let mut ccode_str = ccode.to_string().replace("NEWLINE_TOKEN", "\n");
    ccode_str.push_str(" // FANN_FANN_NET_H_");

    run_cmd!(echo $ccode_str | clang-format > $outfile);
}
