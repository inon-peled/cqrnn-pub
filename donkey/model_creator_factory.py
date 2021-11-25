from create_model import create_model_regularized_lstm128_linear
from create_model import create_model_reg_tanh_lin
from create_model import create_model_reg_stacked_10tanh_lin
from create_model import create_model_reg_10tanh_lin
from create_model import create_model_regularized_lstm_linear
from create_model import create_model_regularized_tanh
from create_model import create_model_lstm_linear
from create_model import create_model_regularized_linear
from create_model import create_model_linear
from create_model import create_model_lstm_linear_multi


def get_model_creator(which):
    return {
        'reg_lstm128_lin': create_model_regularized_lstm128_linear.create_model,
        'reg_lstm_lin': create_model_regularized_lstm_linear.create_model,
        'reg_tanh_lin': create_model_reg_tanh_lin.create_model,
        'reg_stacked_10tanh_lin': create_model_reg_stacked_10tanh_lin.create_model,
        'reg_10tanh_lin': create_model_reg_10tanh_lin.create_model,
        'regularized_tanh': create_model_regularized_tanh.create_model,
        'linear': create_model_linear.create_model,
        'regularized_linear': create_model_regularized_linear.create_model,
        'lstm_linear': create_model_lstm_linear.create_model,
        'lstm_linear_multi': create_model_lstm_linear_multi.create_model
    }[which]
