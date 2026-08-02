import tensorflow as tf
from src.model.tied_lstm import TiedLSTMLanguageModel

model = tf.keras.models.load_model(
    "models/best_model_tied.keras",
    custom_objects={"TiedLSTMLanguageModel": TiedLSTMLanguageModel}
)

print("Loaded OK")
print("Total params:", model.count_params())

# Dummy forward pass to confirm it actually runs, not just loads
dummy_input = tf.zeros((1, model.sequence_length), dtype=tf.int32)
output = model(dummy_input, training=False)
print("Output shape:", output.shape)
print("Vocab size (from model):", model.vocab_size)
print("Row sums to 1:", float(tf.reduce_sum(output, axis=-1).numpy()[0]))