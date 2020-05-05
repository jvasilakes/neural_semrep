import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

X = np.ones((2, 5, 10))
X[0, 1, :] = 2
X[0, 3, :] = 3
X[1, 1, :] = 4
X[1, 3, :] = 3

X_one = X[0]
print("X")
print(X_one)

mask_a_val = tf.constant([True, True, False, False, False])
mask_b_val = tf.constant([False, False, False, True, True])
print(mask_a_val)
print(mask_b_val)

z_a = tf.boolean_mask(X_one, mask_a_val, axis=0)
z_b = tf.boolean_mask(X_one, mask_b_val, axis=0)
print("z_a")
print(z_a)
print("z_b")
print(z_b)

z = tf.stack([z_a, z_b], axis=0)
print("Stacked")
print(z)

summed = tf.reduce_sum(z, axis=1)
print("Summed")
print(summed)

z_2 = tf.expand_dims(tf.reshape(summed, (-1,)), 0)
print("Reshaped")
print(z_2)

dense_layer = tf.keras.layers.Dense(1, name="dense")
output = dense_layer(z_2)
print("Output")
print(tf.reshape(output, (-1, 1)))


print("===============")
print("===============")

input_ = tf.keras.layers.Input(shape=(5, 10), dtype=tf.float32)
print(input_)

mask_a = tf.keras.layers.Input(shape=(5,), dtype=tf.int32)
mask_b = tf.keras.layers.Input(shape=(5,), dtype=tf.int32)

masked_a = tf.boolean_mask(input_, mask_a, axis=0)
print(masked_a)
masked_b = tf.boolean_mask(input_, mask_b, axis=0)
print(masked_b)
masked = tf.stack([masked_a, masked_b], axis=1)
print(masked)

summed = tf.reduce_sum(masked, axis=1)
print(summed)

flat = tf.reshape(summed, (-1, 20))
print(flat)

output = dense_layer(flat)
print(output)
input()

model_inputs = [input_, mask_a, mask_b]
model = tf.keras.models.Model(inputs=model_inputs, outputs=output)
model.compile(loss="binary_crossentropy")
print(model.summary())

print("Output")
mask_a_val = tf.stack([mask_a_val, mask_a_val])
mask_b_val = tf.stack([mask_b_val, mask_b_val])
output = model.predict([X, mask_a_val, mask_b_val])
output = tf.reshape(output, (-1, 1))
print(output)
