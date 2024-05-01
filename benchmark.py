import torch
import click
from functools import partial
import pandas as pd
from kan import KANLayer, KANLayerArguments
from sandbox import get_device
from tqdm import trange
from tabulate import tabulate

# Type mapping for torch data types
__DTYPE__ = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@click.command()
@click.option("--dtype", default="fp32", help="dtype for test [bf16, fp16, fp32]")
@click.option("--steps", default=10, help="number of iterations to run the test for")
@click.option("--warmup_steps", default=5, help="number of iterations to warmup for")
@click.option("--in_features", default=1152, help="in_features arg for perceptron")
@click.option(
    "--out_features", default=1152 * 4, help="out_features arg for perceptron"
)
@click.option(
    "--layer_type", default="kan", help="type of model to benchmark [kan, linear]"
)
@click.option("--batch_size", default=8, help="batch size of input")
def main(dtype, steps, warmup_steps, in_features, out_features, layer_type, batch_size):
    device = get_device()

    # Event class and synchronization function setup depending on device type
    if device.type == "cuda":
        event_class = partial(torch.cuda.Event, enable_timing=True)
        sync = torch.cuda.synchronize
    elif device.type == "mps":
        event_class = partial(torch.mps.event.Event, enable_timing=True)
        sync = torch.mps.synchronize
    else:
        raise NotImplementedError(
            f"No event timer exists for the device type: {device.type}"
        )

    # Creating events for timing
    start_events = [event_class() for _ in range(steps)]
    end_events = [event_class() for _ in range(steps)]

    torch.set_default_dtype(__DTYPE__[dtype])
    torch.set_default_device(device)

    # Initialize model based on type
    if layer_type == "kan":
        args = KANLayerArguments(in_features=in_features, out_features=out_features)
        layer = KANLayer(args)
    elif layer_type == "linear":
        layer = torch.nn.Linear(in_features=in_features, out_features=out_features)
    else:
        raise NotImplementedError(f"{layer_type} is not yet supported")

    # Warm-up phase
    click.secho("Starting warmup...", fg="yellow")
    for _ in trange(warmup_steps, desc="warmup", colour="yellow"):
        x = torch.rand(batch_size, in_features)
        layer(x)

    # Benchmark phase
    click.secho("Starting benchmark...", fg="green")
    for i in trange(steps, desc="benchmark", colour="green"):
        x = torch.rand(batch_size, in_features)
        start_events[i].record()
        layer(x)
        end_events[i].record()

    sync()

    # Collect and print timing data using pandas
    times_df = pd.DataFrame(
        {
            "Step": range(steps),
            "Time (ms)": [
                start.elapsed_time(end) for start, end in zip(start_events, end_events)
            ],
        }
    )

    print(
        tabulate(times_df, headers=times_df.columns, floatfmt="0.3f", tablefmt="grid")
    )
    print(
        f"Average {layer_type} forward pass duration was: {times_df['Time (ms)'].mean():0.2f} ms"
    )
    print(
        f"Standard deviation of {layer_type} forward pass was: {times_df['Time (ms)'].std():0.2f} ms"
    )


if __name__ == "__main__":
    main()
