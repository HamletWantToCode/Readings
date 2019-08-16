import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from autograd import grad
from scipy.optimize import minimize

from KRR import KernelRidge
from local_PCA import Local_PCA
from DFT import Energy, EnergyDer

import time
import os
from OFDFT_ML.utils import pickle_save, read_json, write_json


x_info = []

def info(x):
    global x_info
    x_info.append(x)


def main(config):
    # data
    train_dir = Path(config['data']['train_dir'])
    train_size = config['data']['train_size']
    test_dir = Path(config['data']['test_dir'])
    save_dir = Path(config['results']['save_dir'])
    # model
    gamma = config['model']['KRR']['gamma']
    sigma = config['model']['KRR']['sigma']
    n_components = config['model']['local_pca']['n_components']
    k_nearest = config['model']['local_pca']['k_nearest']

    train_x = np.load(train_dir / 'features.npy')[:train_size]
    train_ys = np.load(train_dir / 'targets.npy')[:train_size]
    train_y, train_dy = train_ys[:, 0], train_ys[:, 1:]
    test_x = np.load(test_dir / 'features.npy')
    test_ys = np.load(test_dir / 'targets.npy')
    test_y, test_dy = test_ys[:, 0], test_ys[:, 1:]

    # fit KRR
    Ek_model = KernelRidge(gamma, sigma, train_x, train_y)
    Ek_model.fit()
    pred_y = Ek_model(test_x)
    Ek_err = np.mean(np.abs(pred_y - test_y))

    # OFDFT
    dens0 = test_x[10]
    target_Vx = -test_dy[35]
    target_dens = test_x[35]

    projector = Local_PCA(n_components, k_nearest, train_x)
    energy = Energy(Ek_model)
    energy_der = EnergyDer(Ek_model, projector)

    ofdft = minimize(energy, dens0, args=(target_Vx),
                method='CG', jac=energy_der, callback=info,
                options={
                    'disp': True,
                })
    pred_dens = ofdft.x
    en_log = [energy(x, target_Vx) for x in x_info]

    config.update({'results': {'Ek_predict_err': Ek_err, 'OFDFT_status': ofdft.success}})
    models = {'KRR': Ek_model, 'local_PCA': projector}
    ground_state = {'init_dens': dens0, 'target_Vx': target_Vx, 'target_dens': target_dens, 'predict_dens': pred_dens}

    datetime = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
    os.makedirs(save_dir / datetime)

    pickle_save(save_dir / datetime / 'models', models)
    pickle_save(save_dir / datetime / 'en_log', en_log)
    pickle_save(save_dir / datetime / 'ground_state_dens', ground_state)
    write_json(save_dir / datetime / 'config.json', config)


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser(description='Machine Learning Training')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args = args.parse_args()
    config = read_json(args.config)
    main(config)
