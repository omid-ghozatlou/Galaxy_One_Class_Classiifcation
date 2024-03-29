import click
import torch
import logging
import random
import numpy as np

from utils.config import Config
from deepSVDD import DeepSVDD
from datasets.galData import load_dataset


################################################################################
# Settings
################################################################################
@click.command()
@click.argument('xp_path',default='D:/Omid/Galaxy-classification-master_0/imgs', type=click.Path(exists=True))
@click.argument('train_test_data_path', default='D:/Omid/Galaxy-classification-master_0/data/Apple/train_test.csv',type=click.Path(exists=True))
@click.argument('apply_model_data_path',default='D:/Omid/Galaxy-classification-master_0/data/Apple/apply_model.csv', type=click.Path(exists=True))
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--objective', type=click.Choice(['one-class', 'soft-boundary']), default='one-class',
              help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
@click.option('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--train', type=bool, default=True,
              help='Train neural network parameters via Deep SVDD.')
@click.option('--optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for Deep SVDD network training.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for Deep SVDD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
@click.option('--pretrain', type=bool, default=True,
              help='Pretrain neural network parameters via autoencoder.')
@click.option('--ae_optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--apply_model', type=bool, default=True,
              help='Model application to unlabelled data.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
              
def main(xp_path, train_test_data_path, apply_model_data_path, load_config=None, load_model=None, objective='one-class', nu=0.1, device='cuda', seed=-1,
         train=True, optimizer_name='adam', lr=0.001, n_epochs=50, lr_milestone=0, batch_size=128, weight_decay=1e-6, pretrain=True, ae_optimizer_name='adam', ae_lr=0.001,
         ae_n_epochs=100, ae_lr_milestone=0, ae_batch_size=128, ae_weight_decay=1e-6, apply_model=True, n_jobs_dataloader=0, normal_class=0):
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('Log file is %s.' % log_file)
    logger.info('Train and test data path is %s.' % train_test_data_path)
    logger.info('Model application data path is %s.' % apply_model_data_path)
    logger.info('Export path is %s.' % xp_path)
    logger.info('Normal class: %d' % normal_class)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print configuration
    logger.info('Deep SVDD objective: %s' % cfg.settings['objective'])
    logger.info('Nu-paramerter: %.2f' % cfg.settings['nu'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset = load_dataset(train_test_data_path, apply_model_data_path, normal_class)   

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
    deep_SVDD.set_network()
    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if load_model:
        deep_SVDD.load_model(model_path=load_model, load_ae=True)
        logger.info('Loading model from %s.' % load_model)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
        logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

        # Pretrain model on dataset (via autoencoder)
        deep_SVDD.pretrain(dataset,
                           optimizer_name=cfg.settings['ae_optimizer_name'],
                           lr=cfg.settings['ae_lr'],
                           n_epochs=cfg.settings['ae_n_epochs'],
                           lr_milestones=cfg.settings['ae_lr_milestone'],
                           batch_size=cfg.settings['ae_batch_size'],
                           weight_decay=cfg.settings['ae_weight_decay'],
                           device=device,
                           n_jobs_dataloader=n_jobs_dataloader)

    logger.info('Training: %s' % train)
    if train:    
        # Log training details
        logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
        logger.info('Training learning rate: %g' % cfg.settings['lr'])
        logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
        logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
        logger.info('Training batch size: %d' % cfg.settings['batch_size'])
        logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

        # Train model on dataset
        deep_SVDD.train(dataset,
                        optimizer_name=cfg.settings['optimizer_name'],
                        lr=cfg.settings['lr'],
                        n_epochs=cfg.settings['n_epochs'],
                        lr_milestones=cfg.settings['lr_milestone'],
                        batch_size=cfg.settings['batch_size'],
                        weight_decay=cfg.settings['weight_decay'],
                        device=device,
                        n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    deep_SVDD.test(export_file=xp_path + '/test_output.txt', dataset=dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # Save results, model, and configuration
    deep_SVDD.save_results(export_json=xp_path + '/results.json')
    deep_SVDD.save_model(export_model=xp_path + '/model.tar')
    cfg.save_config(export_json=xp_path + '/config.json')

    # Apply model and print to file
    if apply_model:
      deep_SVDD.apply_model(export_file=xp_path + '/apply_output.txt', dataset=dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)


if __name__ == '__main__':
    main()
