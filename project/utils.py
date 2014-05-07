import os
import math
import json
import numpy as np
import Image
import cPickle

def make_batch(load_path, file_type, save_path, batch_label, class_list):
  data = []
  file_names = []
  file_list = os.listdir(load_path)
  for item in  file_list:
    if item.endswith("." + file_type):
      path = os.path.join(load_path, item)
      input = Image.open(path)
      img_arr = np.array(input, order='C')
      image = np.fliplr(np.rot90(img_arr, k=3))
      data.append(image.T.flatten('C'))
      file_names.append(item)
  data = np.array(data)
  flip_data = np.flipud(data)
  rotated_data = np.rot90(flip_data, k=3)
  out_file = open(save_path, 'w+')
  
  dic = { 'batch_label': batch_label, 'data': rotated_data, 'labels': class_list, 'filenames': file_names, 'num_cases': len(file_names) }
  cPickle.dump(dic, out_file)
  out_file.close()
  return dic
  
def make_batch_for_class(load_path, file_type, save_path, batch_label, class_num):
  file_names = os.listdir(load_path)
  class_list = [class_num] * len(file_names)
  return make_batch(load_path, file_type, save_path, batch_label, class_list)
  
def unpickle(file_name):
    fo = open(file_name, 'rb')
    dic = cPickle.load(fo)
    fo.close()
    return dic  
  
def show_image(file_name, outpath, img_size, image_num):
  dictionary = unpickle(file_name)
  images = dictionary.get('data')
  
  singleImage = images[:, image_num]

  recon = np.zeros((img_size, img_size, 3), dtype = np.uint8)
  singleImage = singleImage.reshape((img_size * 3, img_size))

  red = singleImage[0:img_size, :]
  blue = singleImage[img_size:2 * img_size, :]
  green = singleImage[2 * img_size:3 * img_size, :]

  recon[:, :, 0] = red
  recon[:, :, 1] = blue
  recon[:, :, 2] = green

  img = Image.fromarray(recon)
  img.save(outpath)
  
def resize_images(width, height, load_path, save_path, file_type):
  filenames = []
  file_list = os.listdir(load_path)
  for item in file_list:
    if item.endswith('.' + file_type):
      n = os.path.join(load_path, item)
      outfile = os.path.join(save_path, item)
      im = Image.open(n)
      im = im.resize((width, height), Image.ANTIALIAS)
      ft = 'JPEG' if file_type.lower() == 'jpg' else file_type
      im.save(outfile, ft)
      
def resize_images_q():
  for i in range(21, 43):
    resize_images(32, 32, "data/class-images/000" + str(i), "data/class-images/000" + str(i), "jpg")
      
def make_batch_from_files(file_names, class_labels, save_path, batch_label):
  data = []
  
  for file_name in file_names:
    input = Image.open(file_name)
    img_arr = np.array(input, order='C')
    image = np.fliplr(np.rot90(img_arr, k=3))
    data.append(image.T.flatten('C'))
    
  data = np.array(data)
  flip_data = np.flipud(data)
  rotated_data = np.rot90(flip_data, k=3)
  out_file = open(save_path, 'w+')
  
  dic = { 'batch_label': batch_label, 'data': rotated_data, 'labels': class_labels, 'filenames': file_names, 'num_cases': len(file_names) }
  cPickle.dump(dic, out_file)
  out_file.close()
  return dic

def make_batch_meta(batch_infos, class_infos, num_cases_per_batch, num_vis, save_path):
  label_names = []
  data_mean = []
  row_datas = []
  
  print 'building up labels'
  label_names = [ clss['label'] for clss in class_infos ]
  
  print 'calculating means (may take a while)'    
  # not sure how they get the mean.. temporary solution
  for row_data in batch_infos[0].get('data'):
    data_mean.append([ sum(row_data) / np.float32(len(row_data)) ])
  data_mean = np.array(data_mean, order='C', dtype=np.float32)
  
  dic = { 'num_cases_per_batch': num_cases_per_batch, 'label_names': label_names, 'num_vis': num_vis, 'data_mean': data_mean }
  
  out_file = open(save_path, 'w+')
  cPickle.dump(dic, out_file)
  out_file.close()
  
def gen_batches_from_config(config_file_name):
  json_data = open(config_file_name)
  info = json.load(json_data)
  json_data.close()
  
  save_path = info.get('save_path')
  files_per_batch = info.get('images_per_batch')
  batch_type = info.get('batch_type')
  num_vis = info.get('num_vis')
  all_classes = info.get('classes')
  
  print 'discovered ' + str(len(all_classes)) + ' classes'
  
  #build up file names for classes
  total_files = 0
  i = 0
  for clss in all_classes:
    clss['id'] = i
    clss['file_names'] = [ os.path.join(clss.get('load_path'), f) for f in os.listdir(clss.get('load_path')) if f.endswith('.' + clss.get('image_type')) ]
    total_files += len(clss.get('file_names'))
    i += 1
  print 'discovered ' + str(total_files) + ' files for batching'  
  
  #generate appropriate batches based on # file names
  total_batches = int(math.floor(total_files / files_per_batch))
  
  #check for unnecessary batch creation
  if total_batches <= 0:
    print 'no batches will be generated'
    return
  else:
    print 'generating ' + str(total_batches) + ' batch files'
  
  print 'clearing out save folder, ' + save_path
  clear_folder(save_path)
  
  #generate batch files
  batch_infos = []
  for b_num in range(1, total_batches + 1):
    print 'generating batch file ' + str(b_num) + ' of ' + str(total_batches)
    #build up file names for this batch
    classes_t = all_classes[:]
    files = []
    labels = []
    curr_class = 0
    #we will take one file from every class until run out/end
    while len(classes_t) > 0 and len(files) < files_per_batch:
      clss = classes_t[curr_class]
      if len(clss.get('file_names')) > 0:
	# get next file name and remove from list
	f_name = clss.get('file_names')[0]
	clss.get('file_names').remove(f_name)
	#append file/label info
	files.append(f_name)
	labels.append(clss.get('id'))
      else:
	# no more files from this class, remove class
	classes_t.remove(clss)
      #select next class
      curr_class = (curr_class + 1) % len(classes_t)
    
    #generate batch from file_names/labels
    batch_info = make_batch_from_files(files, labels, save_path + '/data_batch_' + str(b_num), batch_type + ' batch ' + str(b_num) + ' of ' + str(total_batches))   
    batch_infos.append(batch_info)
  
  #generate batches meta file
  print 'generating batch meta file'
  make_batch_meta(batch_infos, all_classes, files_per_batch, num_vis, save_path + '/batches.meta')
  print 'finished, all generated files saved in ' + info.get('save_path')
  
def clear_folder(path):
  for file_n in os.listdir(path):
    file_path = os.path.join(path, file_n)
    try:
      if os.path.isfile(file_path):
	os.unlink(file_path)
    except Exception, e:
      print e
      
def test_meta(load_path, num_batches):
  batch_infos = []
  class_infos = []
  
  for i in range(1, num_batches + 1):
    batch_infos.append(unpickle(load_path + '/data_batch_' + str(i)))
    
  for i in range(1, 11):
    class_infos.append({ 'label': str(i) })
    
  make_batch_meta(batch_infos, class_infos, 10000, 3072, './batches.meta')
  
  return unpickle('./batches.meta')