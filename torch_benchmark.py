import torch
import yaml
import torchvision

from pytorch_benchmark import benchmark
import pandas as pd
import datetime
import argparse

def test_example(model_name, args):
    ''' Given a model function, test the benchmarking function.
        Returns a pandas.DataFrame with the results.
    '''
    model = getattr(torchvision.models, model_name)()

    if torch.cuda.is_available():
        model = model.cuda()

    sample = torch.randn(args.batch_size, 3, 224, 224)  # (B, C, H, W)

    results = benchmark(model, sample, num_runs=args.num_runs)

    for prop in {"device", "flops", "params", "timing"}:
        assert prop in results

    # print(yaml.dump(results))
    
    # return DataFrame is one column. including rows of each metric
    return pd.DataFrame(
        index = ['model_name', 'params', f'batch={args.batch_size}_latency', f'batches={args.batch_size}_per_second'], 
        data = [model_name, results['params'], 
                results['timing'][f'batch_size_{args.batch_size}']['total']['human_readable']['batch_latency'], 
                results['timing'][f'batch_size_{args.batch_size}']['total']['human_readable']['batches_per_second']]
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_runs', type=int, default=100)
    parser.add_argument('--tests', nargs='+', default=['resnet18', 'resnet101'])
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    all_tests = args.tests
    all_res = []

    for test in all_tests:
        all_res.append(test_example(test, args))

    # concat, save results to csv
    filename = f'./torch_bm-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.csv'

    result = pd.concat(all_res, axis=1)

    result.to_csv(filename)

    print(f'benchmark results saved to {filename}')

    print(result)