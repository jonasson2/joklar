# https://github.com/david8862/tf-keras-deeplabv3p-model-set

from functools import partial
from tensorflow.keras.layers import Conv2D, Reshape, Activation, Softmax, Lambda, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# from deeplabv3p.models.deeplabv3p_xception import Deeplabv3pXception
from deeplabv3p.models.deeplabv3p_mobilenetv2 import Deeplabv3pMobileNetV2, Deeplabv3pLiteMobileNetV2
# from deeplabv3p.models.deeplabv3p_mobilenetv3 import Deeplabv3pMobileNetV3Large, Deeplabv3pLiteMobileNetV3Large, Deeplabv3pMobileNetV3Small, Deeplabv3pLiteMobileNetV3Small
# from deeplabv3p.models.deeplabv3p_mobilevit import Deeplabv3pMobileViT_S, Deeplabv3pLiteMobileViT_S, Deeplabv3pMobileViT_XS, Deeplabv3pLiteMobileViT_XS, Deeplabv3pMobileViT_XXS, Deeplabv3pLiteMobileViT_XXS
# from deeplabv3p.models.deeplabv3p_peleenet import Deeplabv3pPeleeNet, Deeplabv3pLitePeleeNet
# from deeplabv3p.models.deeplabv3p_ghostnet import Deeplabv3pGhostNet, Deeplabv3pLiteGhostNet
from deeplabv3p.models.deeplabv3p_resnet50 import Deeplabv3pResNet50
from deeplabv3p.models.layers import DeeplabConv2D, Subpixel, img_resize

#
# A map of model type to construction function for DeepLabv3+
#
deeplab_model_map = {
    'mobilenetv2': partial(Deeplabv3pMobileNetV2, weights=None, alpha=1.0),
    # 'mobilenetv2_lite': partial(Deeplabv3pLiteMobileNetV2, alpha=1.0),

    # 'mobilenetv3large': partial(Deeplabv3pMobileNetV3Large, alpha=1.0),
    # 'mobilenetv3large_lite': partial(Deeplabv3pLiteMobileNetV3Large, alpha=1.0),

    # 'mobilenetv3small': partial(Deeplabv3pMobileNetV3Small, alpha=1.0),
    # 'mobilenetv3small_lite': partial(Deeplabv3pLiteMobileNetV3Small, alpha=1.0),

    # 'peleenet': Deeplabv3pPeleeNet,
    # 'peleenet_lite': Deeplabv3pLitePeleeNet,

    # 'mobilevit_s': Deeplabv3pMobileViT_S,
    # 'mobilevit_s_lite': Deeplabv3pLiteMobileViT_S,
    # 'mobilevit_xs': Deeplabv3pMobileViT_XS,
    # 'mobilevit_xs_lite': Deeplabv3pLiteMobileViT_XS,
    # 'mobilevit_xxs': Deeplabv3pMobileViT_XXS,
    # 'mobilevit_xxs_lite': Deeplabv3pLiteMobileViT_XXS,

    # 'ghostnet': Deeplabv3pGhostNet,
    # 'ghostnet_lite': Deeplabv3pLiteGhostNet,

    # 'xception': Deeplabv3pXception,
    'resnet50': partial(Deeplabv3pResNet50, weights=None),
}

def get_deeplabv3p_model(model_type, num_classes, model_input_shape, output_stride, freeze_level=0, weights_path=None, training=True, use_subpixel=False):
    # check if model type is valid
    if model_type not in deeplab_model_map.keys():
        raise ValueError('This model type is not supported now')

    model_function = deeplab_model_map[model_type]

    input_tensor = Input(shape=model_input_shape + (13,), name='image_input')
    model, backbone_len = model_function(input_tensor=input_tensor,
                                         input_shape=model_input_shape + (13,),
                                         #weights='imagenet',
                                         num_classes=1,
                                         OS=output_stride)

    base_model = Model(model.input, model.layers[-5].output)
    print('backbone layers number: {}'.format(backbone_len))

    if use_subpixel:
        if model_type == 'xception':
            scale = 4
        else:
            scale = 8
        x = Subpixel(num_classes, 1, scale, padding='same')(base_model.output)
    else:
        x = DeeplabConv2D(num_classes, (1, 1), padding='same', name='conv_upsample')(base_model.output)
        x = Lambda(img_resize, arguments={'size': (model_input_shape[0], model_input_shape[1]), 'mode': 'bilinear'}, name='pred_resize')(x)

    # for training model, we need to flatten mask to calculate loss
    #if training:
    #    x = Reshape((model_input_shape[0]*model_input_shape[1], num_classes)) (x)

    # NOTE: if you want to merge "argmax" postprocess into model,
    #       just switch to following comment code when dumping out
    #       inference model, and remove the np.argmax() action in
    #       postprocess
    # x = Softmax(name='pred_mask')(x)
    import tensorflow as tf
    x = tf.keras.layers.Activation('sigmoid')(x)

    #x = Softmax(name='pred_softmax')(x)
    #if not training:
        #x = Lambda(lambda x: K.argmax(x, axis=-1), name='pred_mask')(x)

    model = Model(base_model.input, x, name='deeplabv3p_'+model_type)

    #if use_subpixel:
        # Do ICNR
        #for layer in model.layers:
            #if type(layer) == Subpixel:
                #c, b = layer.get_weights()
                #w = icnr_weights(scale=scale, shape=c.shape)
                #layer.set_weights([w, b])

    if weights_path:
        model.load_weights(weights_path, by_name=False)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    if freeze_level in [1, 2]:
        # Freeze the backbone part or freeze all but final feature map & input layers.
        num = (backbone_len, len(base_model.layers))[freeze_level-1]
        for i in range(num): model.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(num, len(model.layers)))
    elif freeze_level == 0:
        # Unfreeze all layers.
        for i in range(len(model.layers)):
            model.layers[i].trainable= True
        print('Unfreeze all of the layers.')

    return model
