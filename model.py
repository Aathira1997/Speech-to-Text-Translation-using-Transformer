from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
batch = next(iter(val_ds))

# The vocabulary to convert predicted indices into characters
idx_to_char = vectorizer.get_vocabulary()
display_cb = DisplayOutputs(
    batch, idx_to_char, target_start_token_idx=2, target_end_token_idx=3
)  # set the arguments as per vocabulary index for '<' and '>'


model = Transformer(
    num_hid=200,
    num_head=2,
    num_feed_forward=400,
    target_maxlen=max_target_len,
    num_layers_enc=6,
    num_layers_dec=1,
    num_classes=34,
)
loss_fn = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True, label_smoothing=0.1,
)

learning_rate = 7.071428571428571e-05
optimizer = keras.optimizers.Adam(learning_rate)
model.compile(optimizer=optimizer, loss=loss_fn)

# Define the directory to save the model weights
weights_dir = ''

# Define a ModelCheckpoint callback to save the model weights
checkpoint = ModelCheckpoint(weights_dir + '', save_best_only=True, save_weights_only=True, save_freq='epoch')

# Fit the model and use callbacks to save the model weights and display outputs
history = model.fit(ds, validation_data=val_ds, callbacks=[display_cb, checkpoint], epochs=25)

# Save the entire model in the SavedModel format
tf.saved_model.save(model, weights_dir + '')
