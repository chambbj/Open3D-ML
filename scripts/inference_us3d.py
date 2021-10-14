import argparse
import copy
import os
import open3d.ml as _ml3d
import open3d.ml.tf as ml3d
import pdal
import json
import numpy as np
import yaml
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Predict labels')
    parser.add_argument('input', help='file to predict labels for')
    parser.add_argument('output', help='file with predicted labels')
    parser.add_argument('-c', '--cfg_file', help='path to the config file')
    parser.add_argument('--ckpt_path', help='path to the checkpoint')

    args, unknown = parser.parse_known_args()

    parser_extra = argparse.ArgumentParser(description='Extra arguments')
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser_extra.add_argument(arg)
    args_extra = parser_extra.parse_args(unknown)

    print("regular arguments")
    print(yaml.dump(vars(args)))

    print("extra arguments")
    print(yaml.dump(vars(args_extra)))

    return args, vars(args_extra)

def main():
    cmd_line = ' '.join(sys.argv[:])
    args, extra_dict = parse_args()

    if args.cfg_file is not None:
        # Config file can perhaps default, but we should accept as an arg.
        # cfg_file = "ml3d/configs/randlanet_us3d.yml"
        cfg = _ml3d.utils.Config.load_from_file(args.cfg_file)
        # print("Config\n======")
        # cfg.dataset.cache_dir = "/home/chambbj/code/Open3D-ML/logs/cache_threeclass_irn/"
        # cfg.model.dim_input = 5
        # print(cfg.dataset.cache_dir)
        # print(cfg.model.dim_input)

        # Surely we can determine this by inspecting the config. Is there a more general purpose way to load?
        model = ml3d.models.RandLANet(**cfg.model)
        # print("Model\n=====")
        # print(model.dim_input)
        # model = ml3d.models.KPFCNN(**cfg.model)

        # Could we get away with None? How is it even used in this context? Nope, it's required.
        cfg.dataset['dataset_path'] = "/path/to/dataset/"
        # cfg.dataset['dataset_path'] = None

        # Similarly, this assumes US3D. What happens when we change it? The config tells us what dataset is used.
        dataset = ml3d.datasets.US3D(cfg.dataset.pop('dataset_path', None), **cfg.dataset)

        dataset.cache_dir = cfg.dataset.cache_dir
        model.dim_input = cfg.model.dim_input

        pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)

        # load the parameters.
        if args.ckpt_path is not None:
            # This can be an argument. In practice it's probably just a path to /u02 somewhere.
            pipeline.load_ckpt(ckpt_path=args.ckpt_path)

            # We could modify us3d.py, the US3D dataset, to have a method that reads directly from filename,
            # so that we have one location for modifying how data is loaded. Maybe it's a really good idea to
            # encode the pipeline in the config, so that the same pipeline used to train is used to infer.
            # Regardless, the path should not be hardcoded. It will be provided as a parameter.
            p = pdal.Pipeline(json.dumps([
                args.input,
                # {
                #     "type":"filters.randomize"
                # },
                # {
                #     "type":"filters.sample",
                #     "radius":0.6
                # },
                {
                    "type":"filters.covariancefeatures"
                }
            ]))
            cnt = p.execute()
            # maybe log instead, and provide more information
            print("Processed {} points".format(cnt))

            data = p.arrays[0]
            output = copy.deepcopy(p.arrays[0])
            points = np.vstack((data['X'], data['Y'], data['Z'])).T.astype(np.float32)

            # This bit is ugly. And all the more reason to move the data reading to a central location. Is
            # there a way we can somehow capture the features in the config too? Also, we had to leave off
            # verticality because it broke some assumptions about features in the guts of Open3D.
            feat = np.vstack((data['Linearity'],data['Planarity'],data['Scattering'],data['Verticality'])).T.astype(np.float32)
            # feat = np.vstack((data['X'],data['Y'],data['Z'])).T.astype(np.float32)
            # feat = np.vstack((data['ReturnNumber'],data['Intensity'])).T.astype(np.float32)
            labels = np.zeros((points.shape[0],), dtype=np.int32)

            data = {'point': points, 'feat': feat, 'label': labels}

            # run inference on a single example.
            # returns dict with 'predict_labels' and 'predict_scores'.
            result = pipeline.run_inference(data)
            labels = result['predict_labels']
            output['Classification'] = [dataset.label_values[label] for label in labels]
            # We require predict_labels, but predict_scores as an extra dim could also be quite interesting.
            # output['Scores'] = result['predict_scores']
            dataset.write_result(args.output, output)

if __name__ == '__main__':
    main()