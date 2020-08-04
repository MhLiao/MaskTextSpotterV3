# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# """
# This file contains primitives for multi-gpu communication.
# This is useful when doing distributed training.
# """

# import os
# import pickle
# import tempfile
# import time

# import torch
# import torch.distributed as dist



# # def get_world_size():
# #     if not dist.is_initialized():
# #         return 1
# #     return dist.get_world_size()
# #
# #
# # def is_main_process():
# #     if not dist.is_initialized():
# #         return True
# #     return dist.get_rank() == 0
# #
# # def get_rank():
# #     if not dist.is_initialized():
# #         return 0
# #     return dist.get_rank()
# #
# # def synchronize():
# #     """
# #     Helper function to synchronize between multiple processes when
# #     using distributed training
# #     """
# #     if not dist.is_initialized():
# #         return
# #     world_size = dist.get_world_size()
# #     rank = dist.get_rank()
# #     if world_size == 1:
# #         return
# #
# #     def _send_and_wait(r):
# #         if rank == r:
# #             tensor = torch.tensor(0, device="cuda")
# #         else:
# #             tensor = torch.tensor(1, device="cuda")
# #         dist.broadcast(tensor, r)
# #         while tensor.item() == 1:
# #             time.sleep(1)
# #
# #     _send_and_wait(0)
# #     # now sync on the main process
# #     _send_and_wait(1)
# #
# #
# def _encode(encoded_data, data):
#     # gets a byte representation for the data
#     encoded_bytes = pickle.dumps(data)
#     # convert this byte string into a byte tensor
#     storage = torch.ByteStorage.from_buffer(encoded_bytes)
#     tensor = torch.ByteTensor(storage).to("cuda")
#     # encoding: first byte is the size and then rest is the data
#     s = tensor.numel()
#     assert s <= 255, "Can't encode data greater than 255 bytes"
#     # put the encoded data in encoded_data
#     encoded_data[0] = s
#     encoded_data[1 : (s + 1)] = tensor


# def _decode(encoded_data):
#     size = encoded_data[0]
#     encoded_tensor = encoded_data[1 : (size + 1)].to("cpu")
#     return pickle.loads(bytearray(encoded_tensor.tolist()))


# # TODO try to use tensor in shared-memory instead of serializing to disk
# # this involves getting the all_gather to work
# def scatter_gather(data):
#     """
#     This function gathers data from multiple processes, and returns them
#     in a list, as they were obtained from each process.

#     This function is useful for retrieving data from multiple processes,
#     when launching the code with torch.distributed.launch

#     Note: this function is slow and should not be used in tight loops, i.e.,
#     do not use it in the training loop.

#     Arguments:
#         data: the object to be gathered from multiple processes.
#             It must be serializable

#     Returns:
#         result (list): a list with as many elements as there are processes,
#             where each element i in the list corresponds to the data that was
#             gathered from the process of rank i.
#     """
#     # strategy: the main process creates a temporary directory, and communicates
#     # the location of the temporary directory to all other processes.
#     # each process will then serialize the data to the folder defined by
#     # the main process, and then the main process reads all of the serialized
#     # files and returns them in a list
#     if not dist.is_initialized():
#         return [data]
#     synchronize()
#     # get rank of the current process
#     rank = dist.get_rank()

#     # the data to communicate should be small
#     data_to_communicate = torch.empty(256, dtype=torch.uint8, device="cuda")
#     if rank == 0:
#         # manually creates a temporary directory, that needs to be cleaned
#         # afterwards
#         tmp_dir = tempfile.mkdtemp()
#         _encode(data_to_communicate, tmp_dir)

#     synchronize()
#     # the main process (rank=0) communicates the data to all processes
#     dist.broadcast(data_to_communicate, 0)

#     # get the data that was communicated
#     tmp_dir = _decode(data_to_communicate)

#     # each process serializes to a different file
#     file_template = "file{}.pth"
#     tmp_file = os.path.join(tmp_dir, file_template.format(rank))
#     torch.save(data, tmp_file)

#     # synchronize before loading the data
#     synchronize()

#     # only the master process returns the data
#     if rank == 0:
#         data_list = []
#         world_size = dist.get_world_size()
#         for r in range(world_size):
#             file_path = os.path.join(tmp_dir, file_template.format(r))
#             d = torch.load(file_path)
#             data_list.append(d)
#             # cleanup
#             os.remove(file_path)
#         # cleanup
#         os.rmdir(tmp_dir)
#         return data_list


# def get_world_size():
#     if not dist.is_available():
#         print('distributed is not available')
#         return 1
#     if not dist.is_initialized():
#         print('distributed is not initialized')
#         return 1
#     return dist.get_world_size()


# def get_rank():
#     if not dist.is_available():
#         return 0
#     if not dist.is_initialized():
#         return 0
#     return dist.get_rank()


# def is_main_process():
#     return get_rank() == 0


# def synchronize():
#     """
#     Helper function to synchronize (barrier) among all processes when
#     using distributed training
#     """
#     if not dist.is_available():
#         return
#     if not dist.is_initialized():
#         return
#     world_size = dist.get_world_size()
#     if world_size == 1:
#         return
#     dist.barrier()


# def all_gather(data):
#     """
#     Run all_gather on arbitrary picklable data (not necessarily tensors)

#     Args:
#         data: any picklable object

#     Returns:
#         list[data]: list of data gathered from each rank
#     """
#     world_size = get_world_size()
#     if world_size == 1:
#         return [data]

#     # serialized to a Tensor
#     buffer = pickle.dumps(data)
#     storage = torch.ByteStorage.from_buffer(buffer)
#     tensor = torch.ByteTensor(storage).to("cuda")

#     # obtain Tensor size of each rank
#     local_size = torch.IntTensor([tensor.numel()]).to("cuda")
#     size_list = [torch.IntTensor([0]).to("cuda") for _ in range(world_size)]
#     dist.all_gather(size_list, local_size)
#     size_list = [int(size.item()) for size in size_list]
#     max_size = max(size_list)

#     # receiving Tensor from all ranks
#     # we pad the tensor because torch all_gather does not support
#     # gathering tensors of different shapes
#     tensor_list = []
#     for _ in size_list:
#         tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
#     if local_size != max_size:
#         padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
#         tensor = torch.cat((tensor, padding), dim=0)
#     dist.all_gather(tensor_list, tensor)

#     data_list = []
#     for size, tensor in zip(size_list, tensor_list):
#         buffer = tensor.cpu().numpy().tobytes()[:size]
#         data_list.append(pickle.loads(buffer))

#     return data_list


# def reduce_dict(input_dict, average=True):
#     """
#     Args:
#         input_dict (dict): all the values will be reduced
#         average (bool): whether to do average or sum

#     Reduce the values in the dictionary from all processes so that process with rank
#     0 has the averaged results. Returns a dict with the same fields as
#     input_dict, after reduction.
#     """
#     world_size = get_world_size()
#     if world_size < 2:
#         return input_dict
#     with torch.no_grad():
#         names = []
#         values = []
#         # sort the keys so that they are consistent across processes
#         for k in sorted(input_dict.keys()):
#             names.append(k)
#             values.append(input_dict[k])
#         values = torch.stack(values, dim=0)
#         dist.reduce(values, dst=0)
#         if dist.get_rank() == 0 and average:
#             # only main process gets accumulated, so only divide by
#             # world_size in this case
#             values /= world_size
#         reduced_dict = {k: v for k, v in zip(names, values)}
#     return reduced_dict


"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import pickle
import time

import torch
import torch.distributed as dist


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def scatter_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
