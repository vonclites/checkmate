# checkmate
Checkmate is designed to be a simple drop-in solution for a very common [Tensorflow](https://www.tensorflow.org/) use-case: keeping track of the best model checkpoints during training.


The BestCheckpointSaver is a wrapper around a [tf.train.Saver](https://www.tensorflow.org/api_docs/python/tf/train/Saver).

The BestCheckpointSaver provides the ability to save the **best** *n* checkpoints, whereas the tf.train.Saver can only save the **last** *n* checkpoints.

### Features
* Save only best *n* checkpoints
* Compares checkpoints based on a user-provided value
* Can rank checkpoints by highest or lowest values
* Automatically delete outdated checkpoints
* Provide at a glance record of each checkpoint's associated value (the user-provided value obtained from that checkpoint)

## Using the BestCheckpointSaver
```python
from checkmate import BestCheckpointSaver

# ...build model...

best_ckpt_saver = BestCheckpointSaver(
  save_dir=best_checkpoint_dir,
  num_to_keep=3,
  maximize=True
)

# train and evaluate
for train_step in range(max_steps):
  sess.run(train_op)
  if train_step % evaluation_interval == 0:
    accuracy = sess.run(eval_op, feed_dict=validation_data)
    best_ckpt_saver.handle(accuracy, sess, global_step_tensor)
```

## Loading the best checkpoint
```python
import checkmate

# ...build model...

saver = tf.train.Saver()
saver.restore(sess, checkmate.get_best_checkpoint(best_checkpoint_dir, select_maximum_value=True))
```

At this stage, the module is no-frills with limited documentation.  It is not intended to work in distributed settings or with complex Session/Graph management (i.e. the tf.Estimator framework).  Contributions are welcome.

