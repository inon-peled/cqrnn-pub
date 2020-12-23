from tilted_loss import make_loss_function_tilted_loss
from yu_stander import make_loss_function_yu_standr_nll

loss_function_factory = {
    'tl': make_loss_function_tilted_loss,
    'ys': make_loss_function_yu_standr_nll
}
