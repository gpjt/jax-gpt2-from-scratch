import re
from pathlib import Path

import click

from safetensors.flax import load_file, save_file


MAPPINGS = [
    (r"output_head\.kernel", "out_head.weight", lambda x: x.T),
    (r"output_norm\.bias", "final_norm.shift", lambda x: x.squeeze((0, 1))),
    (r"output_norm\.scale", "final_norm.scale", lambda x: x.squeeze((0, 1))),
    (r"position_embedding\.embedding", "pos_emb.weight", lambda x: x),
    (r"token_embedding\.embedding", "tok_emb.weight", lambda x: x),
    (r"transformers_layers\.layers\.(\d+)\.attention\.W_k\.kernel", "trf_blocks.{group1}.att.W_key.weight", lambda x: x.T),
    (r"transformers_layers\.layers\.(\d+)\.attention\.W_q\.kernel", "trf_blocks.{group1}.att.W_query.weight", lambda x: x.T),
    (r"transformers_layers\.layers\.(\d+)\.attention\.W_v\.kernel", "trf_blocks.{group1}.att.W_value.weight", lambda x: x.T),
    (r"transformers_layers\.layers\.(\d+)\.attention\.output_projection\.kernel", "trf_blocks.{group1}.att.out_proj.weight", lambda x: x.T),
    (r"transformers_layers\.layers\.(\d+)\.attention_norm\.bias", "trf_blocks.{group1}.norm1.shift", lambda x: x.squeeze((0, 1))),
    (r"transformers_layers\.layers\.(\d+)\.attention_norm\.scale", "trf_blocks.{group1}.norm1.scale", lambda x: x.squeeze((0, 1))),
    (r"transformers_layers\.layers\.(\d+)\.ffn\.layers\.0\.bias", "trf_blocks.{group1}.ff.layers.0.bias", lambda x: x),
    (r"transformers_layers\.layers\.(\d+)\.ffn\.layers\.0\.kernel", "trf_blocks.{group1}.ff.layers.0.weight", lambda x: x.T),
    (r"transformers_layers\.layers\.(\d+)\.ffn\.layers\.2\.bias", "trf_blocks.{group1}.ff.layers.2.bias", lambda x: x),
    (r"transformers_layers\.layers\.(\d+)\.ffn\.layers\.2\.kernel", "trf_blocks.{group1}.ff.layers.2.weight", lambda x: x.T),
    (r"transformers_layers\.layers\.(\d+)\.ffn_norm\.scale", "trf_blocks.{group1}.norm2.scale", lambda x: x.squeeze((0, 1))),
    (r"transformers_layers\.layers\.(\d+)\.ffn_norm\.bias", "trf_blocks.{group1}.norm2.shift", lambda x: x.squeeze((0, 1))),
]


def convert(key, value):
    print(key)

    for in_re, out_format, converter in MAPPINGS:
        match = re.compile(in_re).match(key)
        if match is not None:
            print("MATCH")
            if len(match.groups()) == 0:
                result = out_format
            else:
                result = out_format.format(group1=match.group(1))
            print(f"RESULT {result}")
            return result, converter(value)

    raise Exception(key)
    return ""


@click.command()
@click.argument("input_model_safetensors")
@click.argument("output_model_safetensors")
def main(input_model_safetensors, output_model_safetensors):
    if not Path(input_model_safetensors).is_file():
        print(f"{input_model_safetensors} is not a file")
        exit(-1)

    if Path(output_model_safetensors).exists():
        print(f"{output_model_safetensors} already exists")
        exit(-1)

    input_tensors = load_file(input_model_safetensors)

    output_tensors = {}
    for key, tensor in input_tensors.items():
        converted_key, converted_tensor = convert(key, tensor)
        output_tensors[converted_key] = converted_tensor

    save_file(output_tensors, output_model_safetensors)


if __name__ == "__main__":
    main()
