import gc
def get_tensors(only_cuda=False, omit_objs=[]):
    """
    :return: list of active PyTorch tensors
    """
    add_all_tensors = False if only_cuda is True else True
    # To avoid counting the same tensor twice, create a dictionary of tensors,
    # each one identified by its id (the in memory address).
    tensors = {}

    # omit_obj_ids = [id(obj) for obj in omit_objs]

    def add_tensor(obj):
        if torch.is_tensor(obj):
            tensor = obj
            print(tensor.is_cuda)
            print(tensor.shape)

        elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
            tensor = obj.data
            print(tensor.is_cuda)
            print(tensor.shape)

        else:
            return

        if (only_cuda and tensor.is_cuda) or add_all_tensors:
            tensors[id(tensor)] = tensor


    for obj in gc.get_objects():
        try:
            # Add the obj if it is a tensor.
            add_tensor(obj)
            # Some tensors are "saved & hidden" for the backward pass.
            if hasattr(obj, 'saved_tensors') and (id(obj) not in omit_objs):
                for tensor_obj in obj.saved_tensors:
                    add_tensor(tensor_obj)
        except Exception as ex:
            pass
            # print("Exception: ", ex)
            # logger.debug(f"Exception: {str(ex)}")
    return tensors.values()  # return a list of detected tensors