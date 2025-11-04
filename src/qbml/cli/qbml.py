import os
import click
import yaml
from pathlib import Path
import logging

_logger = logging.getLogger(__name__)


@click.group()
def deploy_project():
    pass

@deploy_project.command()
@click.option('-n', '--name', default='qbmlproject', help="Name of the top level directory to deploy QubitML project.")
@click.option('-loc', '--location', default='.', help='Where to build the project structure. Please use the full path.', show_default=True)
def deploy(name, location):
    """
    Deploy the QubitML project structure.
    """
    import qbml.dynamics.deploy as qmldeploy
    qmldeploy.deploy_it(name, location)


@click.group()
def run_dynamics():
    pass

@run_dynamics.command()
@click.option('-c', '--config-pth',  help="Full path to dataset configuration file.", type=click.Path(exists=True))
@click.argument('overrides', nargs=-1)
def tomo(config_pth, overrides):
    """
    Generate qubit tomography datasets.
    """
    cfg_path = Path(config_pth)
    cfg_dir = cfg_path.parent
    cfg_name = cfg_path.stem
    override_string = ""
    for override in overrides:
        override_string += override + " "
    os.system(f'qmltomography -cp {cfg_dir} -cn {cfg_name} {override_string}')

@run_dynamics.command()
@click.option('-cn', help="The name of the config in the setgen directory")
@click.argument('setname', required=True)
@click.argument('overrides', nargs=-1)
def tomo_p(cn, setname, overrides):
    """
    Plot tomography to visualize config.
    """
    override_string = ""
    for override in overrides:
        override_string += override + " "
    os.system(f'qmlviewer -cn {cn} title={setname} {override_string}')

@run_dynamics.command()
@click.option('-cn', help="The name of the config in the setgen directory")
@click.argument('setname', required=True)
@click.argument('overrides', nargs=-1)
def mtomo_p(cn, setname, overrides):
    """
    Plot tomography to visualize config. For (non)Markovian study.
    """
    override_string = ""
    for override in overrides:
        override_string += override + " "
    os.system(f'qmlmviewer -cn {cn} title={setname} {override_string}')


@run_dynamics.command()
@click.option('-cn', help="The name of the config in the setgen directory")
@click.argument('setname', required=True)
@click.argument('setpth', type=click.Path(exists=True))
def tfp(cn, setname, setpth):
    """
    Generate qubit tomography from predicted spectral densities.
    """
    os.system(f'qmltfp -cn {cn} title={setname} ++predictions={setpth}')

@click.group()
def train_models():
    pass

@train_models.command()
@click.option('-c', '--config-pth', help="The name of the config in the training directory.")
@click.argument('overrides', nargs=-1)
def train(config_pth, overrides):
    """
    Train a model.
    """
    cfg_path = Path(config_pth)
    cfg_dir = cfg_path.parent
    cfg_name = cfg_path.stem
    override_string = ""
    for override in overrides:
        override_string += override + " "
    os.system(f'qmltrain -cp {cfg_dir} -cn {cfg_name} {override_string}')


@click.group()
def make_prediction():
    pass

@make_prediction.command()
@click.option("-m/-c", help="Make predictions with (m)odels or (c)ommittees.", default=True)
@click.option("-bs", type=int, help="Prediction batchsize.")
@click.option("-d", "--device", help="Device with which to make predictions.", default='cpu', show_default=True)
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
def predict(m, bs, device, model_path, data_path):
    """
    Make predictions with provided model or committee.
    """
    import qbml.ml.predict as qmlpredict
    qmlpredict.prediction(
        mdlpth=model_path,
        datapth=data_path,
        bs=bs,
        device=device,
        is_model=m,
    )

@click.group()
def compare_configs():
    pass

@compare_configs.command()
@click.argument("cfg_obj", nargs=-1)
def compare(cfg_obj):
    """
    Compare configuration files for data OR models.
    """
    import qbml.ml.compareconfigs as qmlcc
    qmlcc.compareconfigs(cfg_obj)


@click.group()
def assemble_committee():
    pass

@assemble_committee.command()
@click.option("-n", "--name", help="Name for committee directory.")
@click.option("-d", "--device", help="Device for the committee.", default='cpu')
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
def committee(name, device, paths):
    """
    Create a committee from the provided models.
    """
    import qbml.ml.assemblecommittee as qmlcommi
    qmlcommi.assemble_the_committee(name, device, paths)


@click.group()
def config_gen():
    pass

@config_gen.command()
@click.option("-s/-t", default=True, help="(s)etgen or (t)raining.", required=True)
@click.argument("name")
def config(s, name):
    """
    Generate configuration files for setgen and training.
    """
    if s:
        from qbml.cli.selectconfig import setgen_cfg
        cfg = open(f'{name}.yaml', "w")
        yaml.dump(setgen_cfg, cfg)
        click.echo(f"Created {name} configuration!")
    else:
        from qbml.cli.selectconfig import nn_cfg
        cfg = open(f'{name}.yaml', "w")
        yaml.dump(nn_cfg, cfg)
        click.echo(f"Created {name} configuration!")


@click.group()
def augement_sets():
    pass

@augement_sets.command()
@click.argument("newsetname")
@click.option("-p", "--percent", required=True, type=float, help="The standard deviation of the gaussian centered at zero to sample to add error.")
@click.argument("dataset", type=click.Path(exists=True))
def adderror(newsetname, percent, dataset):
    """
    Add error to a dataset.
    """
    from torch import load as tl
    from torch import save as ts
    from qbml.ml.tomographydataset import add_error_to_set
    ds_path = Path("data") / Path(newsetname)
    os.mkdir(ds_path)
    ds = tl(dataset, weights_only=False)
    augmented_set = add_error_to_set(tomography_set=ds,
                                     error_percent=percent)
    parent_set = Path(dataset).stem
    parent_set_pth = ds_path / f'from-{parent_set}.info'
    Path(parent_set_pth).touch()
    ts(augmented_set, ds_path / f'{newsetname}.ds')



@augement_sets.command()
@click.argument("newsetname")
@click.option("-n", "--everynth", required=True, type=int, help="Select every (n)th point from the dynamics in the dataset.")
@click.argument("dataset", type=click.Path(exists=True))
def sparsify(newsetname, everynth, dataset):
    """
    Remove every nth time point from dynamics in dataset.
    """
    from torch import load as tl
    from torch import save as ts
    from qbml.ml.tomographydataset import sparsify_set
    ds_path = Path("data") / Path(newsetname)
    os.mkdir(ds_path)
    ds = tl(dataset, weights_only=False)
    augmented_set = sparsify_set(tomography_set=ds,
                                 every_nth=everynth)
    parent_set = Path(dataset).stem
    parent_set_pth = ds_path / f'from-{parent_set}.info'
    Path(parent_set_pth).touch()
    ts(augmented_set, ds_path / f'{newsetname}.ds')

@augement_sets.command()
@click.argument("newsetname")
@click.option("-s", "--scale", required=True, type=float, help="Select every (n)th point from the dynamics in the dataset.")
@click.argument("dataset", type=click.Path(exists=True))
def scale(newsetname, scale, dataset):
    """
    Scale the spectral density values at each frequency.
    """
    from torch import load as tl
    from torch import save as ts
    from qbml.ml.tomographydataset import scale_set_specdens
    ds_path = Path("data") / Path(newsetname)
    os.mkdir(ds_path)
    ds = tl(dataset, weights_only=False)
    augmented_set = scale_set_specdens(tomography_set=ds,
                                       scale_factor=scale)
    parent_set = Path(dataset).stem
    parent_set_pth = ds_path / f'from-{parent_set}.info'
    Path(parent_set_pth).touch()
    ts(augmented_set, ds_path / f'{newsetname}.ds')


@click.group
def set_combo():
    pass

@set_combo.command()
@click.option("-n", "--setname", required=True, type=str, help="Name of the new dataset.")
@click.argument("setpaths", nargs=-1, type=click.Path(exists=True))
def combine(setname, setpaths):
    """
    Combine two datasets
    """
    from torch import load as tload
    from torch import save as tsave
    from qbml.ml.tomographydataset import combine_tomos
    ds_path = Path("data") / Path(setname)
    os.mkdir(ds_path)
    parent_sets = Path(setpaths[0]).stem
    current_set = tload(setpaths[0], weights_only=False)
    remaining_sets = setpaths[1:]
    for tomopth in remaining_sets:
        tmp_set = tload(tomopth, weights_only=False)
        parent_sets += Path(tomopth).stem
        current_set = combine_tomos(current_set, tmp_set)
    parent_set_pth = ds_path / f'from-{parent_sets}.info'
    Path(parent_set_pth).touch()
    tsave(current_set, ds_path / f'{setname}.ds')





cli = click.CommandCollection(sources=[deploy_project, run_dynamics, make_prediction, train_models, compare_configs, assemble_committee, config_gen, augement_sets, set_combo])

if __name__ == "__main__":
    cli()
