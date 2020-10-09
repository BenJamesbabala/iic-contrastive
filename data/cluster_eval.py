import sys
import datetime
import numpy as np
import torch
from simclr_clustering.archs.loss import IICLoss
from .eval_metrics import _hungarian_match, _original_match, _acc
from .transforms import sobel_process

#Get flat predictions and targets from a single dataloaders
#Targets are a single list, predictions are a list per subhead
def _clustering_get_data(config, net, dataloader, sobel=False, using_IR=False, get_soft=False, verbose=0):
  """
  Given a dataloader, returns flat prediction and ground truth, and optionally softmax assignments. Used only for testing.
  """
  num_batches = len(dataloader)
  #Creates tensors for predictions and targets. Prediction are built for each sub-head
  flat_targets_all = torch.zeros((num_batches * config.iic_bs), dtype=torch.int32).cuda()
  flat_predss_all = [torch.zeros((num_batches * config.iic_bs), dtype=torch.int32).cuda() for _ in range(config.num_sub_heads)]
  #Also create softmax prediction if requested
  if get_soft:
    soft_predss_all = [torch.zeros((num_batches * config.iic_bs, config.output_k_B), dtype=torch.float32).cuda() for _ in range(config.num_sub_heads)]
  num_test = 0
  #Iterating batches
  for b_i, batch in enumerate(dataloader):
    imgs = batch[0].cuda()
    #Transform is necessary and get ground truth
    if sobel:
      imgs = sobel_process(imgs, config.include_rgb, using_IR=using_IR)
    flat_targets = batch[1]
    #Net returns a list of predictions, one for each head
    with torch.no_grad():
      x_outs = net(imgs)
    #Shape is (batch_size, output_k)
    assert (x_outs[0].shape[1] == config.output_k_B)
    assert (len(x_outs[0].shape) == 2)
    #Length of current batch
    num_test_curr = flat_targets.shape[0]
    num_test += num_test_curr

    #Iterates every head result
    start_i = b_i * config.iic_bs
    for i in range(config.num_sub_heads):
      x_outs_curr = x_outs[i]
      flat_preds_curr = torch.argmax(x_outs_curr, dim=1)  # along output_k
      #Flat prediction of i subhead of a batch
      flat_predss_all[i][start_i:(start_i + num_test_curr)] = flat_preds_curr
      #Same for soft prediction
      if get_soft:
        soft_predss_all[i][start_i:(start_i + num_test_curr), :] = x_outs_curr
    #Same for targets
    flat_targets_all[start_i:(start_i + num_test_curr)] = flat_targets.cuda()
  #At the end the lists are linearized
  flat_predss_all = [flat_predss_all[i][:num_test] for i in range(config.num_sub_heads)]
  flat_targets_all = flat_targets_all[:num_test]

  if not get_soft:
    return flat_predss_all, flat_targets_all
  else:
    soft_predss_all = [soft_predss_all[i][:num_test] for i in range(config.num_sub_heads)]
    return flat_predss_all, flat_targets_all, soft_predss_all

#Returns general evalutation metrics
def cluster_subheads_eval(config, net, mapping_assignment_dataloader, mapping_test_dataloader, sobel, using_IR=False, get_data_fn=_clustering_get_data, use_sub_head=None, verbose=0):
  """
  Used by both clustering and segmentation.
  Returns metrics for test set.
  Get result from average accuracy of all sub_heads (mean and std).
  All matches are made from training data.
  Best head metric, which is order selective unlike mean/std, is taken from 
  best head determined by training data (but metric computed on test data).
  
  Option to choose best sub_head either based on loss (set use_head in main 
  script), or eval. Former does not use labels for the selection at all and this
  has negligible impact on accuracy metric for our models.
  """

  all_matches, train_accs = _get_assignment_data_matches(net,
                                                         mapping_assignment_dataloader,
                                                         config,
                                                         sobel=sobel,
                                                         using_IR=using_IR,
                                                         get_data_fn=get_data_fn,
                                                         verbose=verbose)
  #get best head based on accuracy, or a specific head from config
  best_sub_head_eval = np.argmax(train_accs)
  if (config.num_sub_heads > 1) and (use_sub_head is not None):
    best_sub_head = use_sub_head
  else:
    best_sub_head = best_sub_head_eval

  test_accs = train_accs
  #returns a dict of eval metrics
  return {"test_accs": list(test_accs),
          "avg": np.mean(test_accs),
          "std": np.std(test_accs),
          "best": test_accs[best_sub_head],
          "worst": test_accs.min(),
          "best_train_sub_head": best_sub_head,  # from training data
          "best_train_sub_head_match": all_matches[best_sub_head],
          "train_accs": list(train_accs)}

#Returns for each matches between ground truth clusters and predictions, and optionally accuracy
def _get_assignment_data_matches(net, mapping_assignment_dataloader, config, sobel=False, using_IR=False, get_data_fn=None, just_matches=False, verbose=0):
  """
  Get all best matches per head based on train set i.e. mapping_assign,
  and mapping_assign accs.
  """

  #get_data_fn is a function returning prediction and targets of all dataset
  flat_predss_all, flat_targets_all = get_data_fn(config, net, mapping_assignment_dataloader, sobel=sobel, using_IR=using_IR, verbose=verbose)
  #flat_predss_all is a list of a sublist for each subhead. Each sublist contains flat prediction of all dataset
  num_test = flat_targets_all.shape[0]
  #Should be equal, both are the number of samples in dataset
  assert (flat_predss_all[0].shape == flat_targets_all.shape)
  num_samples = flat_targets_all.shape[0]

  all_matches = []
  if not just_matches:
    #Prepare an array for calculating accuracy per subhead
    all_accs = np.zeros(config.num_sub_heads, dtype=np.float32)

  #Calculates matches for each subhead with the chosen match algorithm  
  for i in range(config.num_sub_heads):

    if config.eval_mode == "hung":
      match = _hungarian_match(flat_predss_all[i], flat_targets_all, preds_k=config.output_k_B, targets_k=config.gt_k)
    elif config.eval_mode == "orig":
      match = _original_match(flat_predss_all[i], flat_targets_all, preds_k=config.output_k_B, targets_k=config.gt_k)
    else:
      assert (False)
    all_matches.append(match)

    #Needed to return also accuracy of each head
    if not just_matches:
      # reorder predictions to be same cluster assignments as gt_k
      found = torch.zeros(config.output_k_B)
      reordered_preds = torch.zeros(num_samples, dtype=flat_predss_all[0].dtype).cuda()
      for pred_i, target_i in match:
        reordered_preds[flat_predss_all[i] == pred_i] = target_i
        found[pred_i] = 1
      assert (found.sum() == config.output_k_B)  # each output_k must get mapped

      acc = _acc(reordered_preds, flat_targets_all, config.gt_k, verbose)
      all_accs[i] = acc
  #for each head assignment (and optionally accuracy) are returned
  if just_matches:
    return all_matches
  else:
    return all_matches, all_accs

#Get performance of single subhead by loss instead of evalutation metric
def get_subhead_using_loss(config, dataloaders_head_B, net, sobel, lamb, compare=False):
  net.eval()

  head = "B"  # main output head
  dataloaders = dataloaders_head_B
  iterators = (d for d in dataloaders)

  b_i = 0
  loss_per_sub_head = np.zeros(config.num_sub_heads)
  for tup in zip(*iterators):
    net.module.zero_grad()
    dim = config.in_channels
    if sobel:
      dim -= 1
    all_imgs = torch.zeros(config.iic_bs, dim, config.input_sz, config.input_sz).cuda()
    all_imgs_tf = torch.zeros(config.iic_bs, dim, config.input_sz, config.input_sz).cuda()

    imgs_curr = tup[0][0]  # always the first
    curr_iic_bs = imgs_curr.size(0)
    for d_i in range(config.num_dataloaders):
      imgs_tf_curr = tup[1 + d_i][0]  # from 2nd to last
      assert (curr_iic_bs == imgs_tf_curr.size(0))

      actual_batch_start = d_i * curr_iic_bs
      actual_batch_end = actual_batch_start + curr_iic_bs
      all_imgs[actual_batch_start:actual_batch_end, :, :, :] = imgs_curr.cuda()
      all_imgs_tf[actual_batch_start:actual_batch_end, :, :, :] = imgs_tf_curr.cuda()

    curr_total_iic_bs = curr_iic_bs * config.num_dataloaders
    all_imgs = all_imgs[:curr_total_iic_bs, :, :, :]
    all_imgs_tf = all_imgs_tf[:curr_total_iic_bs, :, :, :]

    if sobel:
      all_imgs = sobel_process(all_imgs, config.include_rgb)
      all_imgs_tf = sobel_process(all_imgs_tf, config.include_rgb)

    with torch.no_grad():
      x_outs = net(all_imgs, head=head)
      x_tf_outs = net(all_imgs_tf, head=head)

    for i in range(config.num_sub_heads):
      loss, loss_no_lamb = IID_loss(x_outs[i], x_tf_outs[i], lamb=lamb)
      loss_per_sub_head[i] += loss.item()

    if b_i % 100 == 0:
      print("at batch %d" % b_i)
      sys.stdout.flush()
    b_i += 1

  best_sub_head_loss = np.argmin(loss_per_sub_head)

  if compare:
    print(loss_per_sub_head)
    print("best sub_head by loss: %d" % best_sub_head_loss)

    best_epoch = np.argmax(np.array(config.epoch_acc))
    if "best_train_sub_head" in config.epoch_stats[best_epoch]:
      best_sub_head_eval = config.epoch_stats[best_epoch]["best_train_sub_head"]
      test_accs = config.epoch_stats[best_epoch]["test_accs"]
    else:  # older config version
      best_sub_head_eval = config.epoch_stats[best_epoch]["best_head"]
      test_accs = config.epoch_stats[best_epoch]["all"]

    print("best sub_head by eval: %d" % best_sub_head_eval)

    print("... loss select acc: %f, eval select acc: %f" % (test_accs[best_sub_head_loss], test_accs[best_sub_head_eval]))
  net.train()
  return best_sub_head_loss

def cluster_eval(config, net, mapping_assignment_dataloader, mapping_test_dataloader, sobel, use_sub_head=None):

  net.eval()
  stats_dict = cluster_subheads_eval(config, net,
                                     mapping_assignment_dataloader=mapping_assignment_dataloader,
                                     mapping_test_dataloader=mapping_test_dataloader,
                                     sobel=sobel,
                                     use_sub_head=use_sub_head)
  net.train()
  return (stats_dict['avg'], stats_dict['best'])