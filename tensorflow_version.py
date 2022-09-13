import tensorflow as tf
import time
import random

# Creates some virtual devices (cpu:0, cpu:1, etc.) for using distribution strategy
physical_devices = tf.config.list_physical_devices("CPU")
tf.config.experimental.set_virtual_device_configuration(
    physical_devices[0], [
        tf.config.experimental.VirtualDeviceConfiguration(),
        tf.config.experimental.VirtualDeviceConfiguration(),
        tf.config.experimental.VirtualDeviceConfiguration()
    ])

nval = 3111696*5
start_tensorflow = time.time()
g1 = tf.random.Generator.from_seed(1)
g1.normal(shape=[1, nval])
end_tensorflow = time.time()

print(end_tensorflow-start_tensorflow)

start_random = time.time()
nval = 3111696*5
for el in range(nval):
    random.random()
end_random = time.time()
print(end_random-start_random)