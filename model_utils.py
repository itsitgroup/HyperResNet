from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, UpSampling2D
from keras.layers import BatchNormalization, Dropout, LeakyReLU, Add, Dense, Flatten, Reshape, Multiply
from keras.optimizers import Adam

def create_residual_block(input_layer, num_filters, kernel_size=(3, 3)):
    x = Conv2D(num_filters, kernel_size, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(num_filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    return Add()([input_layer, x])

def create_attention_block(input_layer, num_filters):
    avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(num_filters // 8, activation='relu')(avg_pool)
    dense2 = Dense(num_filters, activation='sigmoid')(dense1)
    scale = Multiply()([input_layer, Reshape((1, 1, num_filters))(dense2)])
    return scale

def encoder(input_layer, num_filters, num_blocks):
    x = Conv2D(num_filters, (3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    for _ in range(num_blocks):
        x = create_residual_block(x, num_filters)
        x = create_attention_block(x, num_filters)
    
    x = MaxPooling2D(pool_size=(2, 2))(x)
    return x

def decoder(input_layer, num_filters, num_blocks):
    x = UpSampling2D(size=(2, 2))(input_layer)
    
    for _ in range(num_blocks):
        x = create_residual_block(x, num_filters)
        x = create_attention_block(x, num_filters)
    
    x = Conv2DTranspose(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x

def create_model():
    hi_res_rgb_input = Input(shape=(64, 64, 3))
    low_res_hsi_input = Input(shape=(8, 8, 31))

    rgb_branch = encoder(hi_res_rgb_input, 64, 3)
    upsampled_hsi_branch = UpSampling2D(size=(8, 8))(low_res_hsi_input)
    hsi_branch = encoder(upsampled_hsi_branch, 64, 3)
    
    fused = concatenate([rgb_branch, hsi_branch])
    
    decoder_output = decoder(fused, 64, 3)
    output = Conv2D(31, (3, 3), activation='sigmoid', padding='same')(decoder_output)

    model = Model(inputs=[hi_res_rgb_input, low_res_hsi_input], outputs=[output])
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    return model
