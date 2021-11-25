import create_model_reg_tanh_lin
import create_model_reg_stacked_10tanh_lin
import create_model_reg_10tanh_lin
import create_model_regularized_lstm_linear
import create_model_regularized_tanh
import create_model_lstm_linear
import create_model_regularized_linear
import create_model_linear
import create_model_lstm_linear_multi
import create_model_regularized_lstm_linear_multi
import create_model_linear_multi
import create_model_regularized_linear_multi


model_creator_factory = {
    'reg_tanh_lin': create_model_reg_tanh_lin.create_model,
    'reg_stacked_10tanh_lin': create_model_reg_stacked_10tanh_lin.create_model,
    'reg_10tanh_lin': create_model_reg_10tanh_lin.create_model,
    'regularized_lstm_linear': create_model_regularized_lstm_linear.create_model,
    'regularized_tanh': create_model_regularized_tanh.create_model,
    'linear': create_model_linear.create_model,
    'regularized_linear': create_model_regularized_linear.create_model,
    'lstm_linear': create_model_lstm_linear.create_model,
    'regularized_lstm_linear_multi': create_model_regularized_lstm_linear_multi.create_model,
    'lstm_linear_multi': create_model_lstm_linear_multi.create_model,
    'linear_multi': create_model_linear_multi.create_model,
    'linear_reg_multi': create_model_regularized_linear_multi.create_model,
}
