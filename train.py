import os
import random
import numpy as np
import tensorflow as tf
from metrics import Progbar
from config import Config
from data_load import Dataset
from model import GDNInpainting


def main():
    config_path = os.path.join('config.yml')
    config = Config(config_path)
    config.print()

    # Init cuda environment
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

    # Init random seed to less result random
    tf.set_random_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # Init training data
    with tf.variable_scope('input_data'):
        dataset = Dataset(config)
        batch_img = dataset.batch_image
        batch_mask = dataset.batch_mask
        val_img = dataset.val_image
        val_mask = dataset.val_mask

    # Init the model
    model = GDNInpainting(config)

    # Build train model
    gen_loss, dis_loss, psnr = model.build_whole_model(batch_img, batch_mask)
    gen_optim, dis_optim = model.build_optim(gen_loss, dis_loss)

    # Build validate model
    val_weighted_loss, val_l1_loss, val_dis_loss, val_psnr = model.build_validation_model(val_img, val_mask)

    # Create the graph
    config_graph = tf.ConfigProto()
    config_graph.gpu_options.allow_growth = True
    with tf.Session(config=config_graph) as sess:
        # Merge all the summaries
        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter(config.CHECKPOINTS + 'train', sess.graph)
        eval_writer = tf.summary.FileWriter(config.CHECKPOINTS + 'eval')

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if config.LOAD_MODEL is not None:
            checkpoint = tf.train.get_checkpoint_state(config.LOAD_MODEL)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(config.LOAD_MODEL))
            step = int(meta_graph_path.split("-")[2].split(".")[0]) * (dataset.len_train / dataset.batch_size)
        else:
            step = 0

        # Start input enqueue threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        progbar = Progbar(dataset.len_train, width=20, stateful_metrics=['epoch', 'iter', 'gen_loss', 'dis_loss', 'psnr'])
        try:
            while not coord.should_stop():
                step += 1
                epoch = int(step / dataset.len_train * dataset.batch_size)
                g_loss, d_loss, t_psnr, _, _ = sess.run([gen_loss, dis_loss, psnr, gen_optim, dis_optim])

                logs = [
                    ("epoch", epoch),
                    ("iter", step),
                    ("gen_loss", g_loss),
                    ("dis_loss", d_loss),
                    ("psnr", t_psnr),
                ]
                progbar.add(dataset.batch_size, values=logs)

                if step % (dataset.len_train / dataset.batch_size) == 0:
                    progbar = Progbar(dataset.len_train, width=20,
                                      stateful_metrics=['epoch', 'iter', 'gen_loss', 'dis_loss', ])

                if (step + 1) % config.SUMMARY_INTERVAL == 0:
                    # Run validation
                    v_psnr = []
                    w_loss = []
                    l1_loss = []
                    dd_loss = []

                    for i in range(dataset.len_val // dataset.val_batch_size):
                        ts_psnr, ts_weighted_loss, ts_l1_loss, ts_dd_loss = sess.run(
                            [val_psnr, val_weighted_loss, val_l1_loss, val_dis_loss])
                        v_psnr.append(ts_psnr)
                        w_loss.append(ts_weighted_loss)
                        l1_loss.append(ts_l1_loss)
                        dd_loss.append(ts_dd_loss)

                    eval_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='psnr', simple_value=np.mean(v_psnr))]), epoch)
                    eval_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='loss/gen_weighted_loss', simple_value=np.mean(w_loss))]),
                        epoch)
                    eval_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='loss/gen_l1_loss', simple_value=np.mean(l1_loss))]),
                        epoch)
                    eval_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='loss/dis_loss', simple_value=np.mean(dd_loss))]), epoch)

                    # Train summary
                    summary = sess.run(merged)
                    train_writer.add_summary(summary, epoch)

                if (step + 1) % config.SAVE_INTERVAL == 0:
                    saver.save(sess, config.CHECKPOINTS + 'log', global_step=epoch, write_meta_graph=False)


        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        # Wait for threads to finish
        coord.join(threads)
    sess.close()


if __name__ == "__main__":
    main()
