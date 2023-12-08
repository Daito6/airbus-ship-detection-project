from keras import backend as K

from defines import ALPHA, GAMMA


def Loss(targets, inputs, alpha=ALPHA, gamma=GAMMA):
    """Custom Loss Function"""
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    bce = K.binary_crossentropy(targets, inputs)
    bce_exp = K.exp(-bce)
    loss = K.mean(alpha * K.pow((1 - bce_exp), gamma) * bce)

    return loss


def dice_coef(y_true, y_pred, smooth=1):
    """Dice Coefficient Metric"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ((2. * intersection + smooth) /
            (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
