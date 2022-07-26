import argparse

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(path_to_onnx, engine_cache_path):
    builder = trt.Builder(TRT_LOGGER)

    config = builder.create_builder_config()
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.max_workspace_size = 5 * 1024 * 1024 * 1024
    config.set_flag(trt.BuilderFlag.INT8)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, TRT_LOGGER)
    onnx_parser.parse_from_file(path_to_onnx)

    engine = builder.build_engine(network, config)
    with open(engine_cache_path, 'wb') as engine_cache:
        engine_cache.write(engine.serialize())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compile engine from onnx file.')
    parser.add_argument('--onnx_model', type=str, required=True, help='path to onnx')
    parser.add_argument('--output', type=str, required=True, help='path to output engine file.')

    args = parser.parse_args()
    print('Compile')
    build_engine(args.onnx_model, args.output)
