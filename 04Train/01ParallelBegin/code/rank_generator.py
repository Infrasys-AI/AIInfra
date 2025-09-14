from typing import List, Optional, Callable
import argparse


def generate_masked_orthogonal_rank_groups(
    world_size: int, parallel_size: List[int], mask: List[bool]
) -> List[List[int]]:
    r"""Generate orthogonal parallel groups based on the parallel size and mask.

    Arguments:
        world_size (int): world size

        parallel_size (List[int]):
            The parallel size of each orthogonal parallel type. For example, if
            tensor_parallel_size = 2, pipeline_model_parallel_group = 3, data_parallel_size = 4,
            and the parallel mapping order is tp-pp-dp, then the parallel_size = [2, 3, 4].

        mask (List[bool]):
            The mask controls which parallel methods the generated groups represent. If mask[i] is
            True, it means the generated group contains the i-th parallelism method. For example,
            if parallel_size = [tp_size, pp_size, dp_size], and mask = [True, False , True], then
            the generated group is the `tp-dp` group, if the mask = [False, True, False], then the
            generated group is the `pp` group.

    Algorithm:
        For orthogonal parallelism, such as tp/dp/pp/cp, the global_rank and
        local_rank satisfy the following equation:
            global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size (1)
                tp_rank \in [0, tp_size)
                dp_rank \in [0, dp_size)
                pp_rank \in [0, pp_size)

        If we want to get the `dp_group` (tp_size * pp_size groups of dp_size ranks each.
        For example,  if the gpu size is 8 and order is 'tp-pp-dp', size is '2-2-2', and the
        dp_group here is [[0, 4], [1, 5], [2, 6], [3, 7]].)
        The tp_rank and pp_rank will be combined to form the `dp_group_index`.
            dp_group_index = tp_rank + pp_rank * tp_size (2)

        So, Given that tp_rank and pp_rank satisfy equation (2), and dp_rank in
        range(0, dp_size), the ranks in dp_group[dp_group_index] satisfies the
        equation (1).

        This function solve this math problem.

    For example, if the parallel_size = [tp_size, dp_size, pp_size] = [2, 3, 4],
    and the mask = [False, True, False]. Then,
        dp_group_index(0) = tp_rank(0) + pp_rank(0) * 2
        dp_group_index(1) = tp_rank(1) + pp_rank(0) * 2
        ...
        dp_group_index(7) = tp_rank(1) + pp_rank(3) * 2

        dp_group[0] = 0 + range(0, 3) * 2 + 0 = [0, 2, 4]
        dp_group[1] = 1 + range(0, 3) * 2 + 0 = [1, 3, 5]
        ...
        dp_group[7] = 1 + range(0, 3) * 2 + 3 * 2 * 3 = [19, 21, 23]
    """

    def prefix_product(a: List[int], init=1) -> List[int]:
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r

    def inner_product(a: List[int], b: List[int]) -> int:
        return sum([x * y for x, y in zip(a, b)])

    def decompose(index, shape, stride=None):
        """
        This function solve the math problem below:
            There is an equation:
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will be used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        """
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        assert (
            sum([x * y for x, y in zip(idx, stride[:-1])]) == index
        ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size

    ranks = []
    for group_index in range(num_of_group):
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        for rank_in_group in range(group_size):
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride)
                + inner_product(decomposed_group_idx, unmasked_stride)
            )
        ranks.append(rank)
    return ranks


class RankGenerator(object):
    """A class for generating rank groups for different modes of parallelism."""

    def __init__(
        self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str, rank_offset: int = 0
    ) -> None:
        assert (
            ep == 1 or cp == 1
        ), "Both EP and CP > 1 in not allow in one rank generator. \
            CP is only included in default RankGenerator, and EP only in expert RankGenerator."

        self.tp = tp
        self.ep = ep
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.rank_offset = rank_offset
        self.world_size = tp * dp * pp * cp * ep

        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "cp": self.cp,
        }
        self.order = order
        order = order.lower()

        for name in self.name_to_size.keys():
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't"
                    f"specified the order ({self.order})."
                )
            elif name not in order:
                order = order + '-' + name

        self.order = order
        self.ordered_size = []

        for token in order.split('-'):
            self.ordered_size.append(self.name_to_size[token])

    def get_mask(self, order: str, token: str):
        """Create a mask for the specified tokens based on the given order.

        Args:
            order (str): The order of parallelism types (e.g., 'tp-dp-pp').
            token (str): The specific parallelism types to include in the mask,
                         separated by hyphens (e.g., 'tp-dp').
        """
        ordered_token = order.split('-')
        token_list = token.split('-')
        mask = [False] * len(ordered_token)
        for t in token_list:
            mask[ordered_token.index(t)] = True
        return mask

    def get_ranks(self, token):
        """Get rank group by input token.

        Args:
            token (str):
                Specify the ranks type that want to get. If we want
                to obtain multiple parallel types, we can use a hyphen
                '-' to separate them. For example, if we want to obtain
                the TP_DP group, the token should be 'tp-dp'.
        """
        mask = self.get_mask(self.order, token)
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, self.ordered_size, mask)
        if self.rank_offset > 0:
            for rank_group in ranks:
                for i in range(len(rank_group)):
                    rank_group[i] += self.rank_offset
        return ranks


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: Optional[int] = None,
    order: str = "tp-cp-ep-dp-pp",
    world_size: int = 1,
) -> None:
    """Initialize model data parallel groups.

    Args:
        tensor_model_parallel_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pipeline_model_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if
            tensor_model_parallel_size is 4 and
            pipeline_model_parallel_size is 2, the model will be split
            into 2 groups of 4 GPUs.

        virtual_pipeline_model_parallel_size (int, optional):
            The number of stages that each pipeline group will have,
            interleaving as necessary. If None, no interleaving is
            performed. For example, if tensor_model_parallel_size is 1,
            pipeline_model_parallel_size is 4,
            virtual_pipeline_model_parallel_size is 2, and there are
            16 transformer layers in the model, the model will be
            split into 8 stages with two layers each and each GPU
            would get 2 stages as such (layer number starting with 1):

            GPU 0: [1, 2] [9, 10]
            GPU 1: [3, 4] [11, 12]
            GPU 2: [5, 6] [13, 14]
            GPU 3: [7, 8] [15, 16]

        context_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            network input sequence length across. Compute of attention
            module requires tokens of full sequence length, so GPUs
            in a context parallel group need to communicate with each
            other to exchange information of other sequence chunks.
            Each GPU and its counterparts in other tensor parallel
            groups compose a context parallel group.

            For example, assume we have 8 GPUs, if tensor model parallel
            size is 4 and context parallel size is 2, the network input
            will be split into two sequence chunks, which are processed
            by 2 different groups of 4 GPUs. One chunk is processed by
            GPU0-3, the other chunk is processed by GPU4-7. Four groups
            are build to do context parallel communications: [GPU0, GPU4],
            [GPU1, GPU5], [GPU2, GPU6], and [GPU3, GPU7].

            Context parallelism partitions sequence length, so it has no
            impact on weights, which means weights are duplicated among
            GPUs in a context parallel group. Hence, weight gradients
            all-reduce is required in backward. For simplicity, we piggyback
            GPUs of context parallelism on data parallel group for
            weight gradient all-reduce.

        expert_model_parallel_size (int, default = 1):
            The number of Mixture of Experts parallel GPUs in each expert
            parallel group.

        expert_tensor_parallel_size (int, default = tp_size):
            The number of GPUs to split individual tensors of expert.

        order (str, default=tp-dp-pp):
            The rank initialization order of parallelism. Now we support
            tp-dp-pp and tp-pp-dp orders.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.

    """

    decoder_model_size = (
        tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )
    total_model_size = decoder_model_size

    if world_size % total_model_size != 0:
        raise RuntimeError(f"world_size ({world_size}) is not divisible by {total_model_size}")

    data_parallel_size: int = world_size // total_model_size

    decoder_world_size = decoder_model_size * data_parallel_size

    
    encoder_rank_generator = None

    decoder_rank_generator = RankGenerator(
        tp=tensor_model_parallel_size,
        ep=1,
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=context_parallel_size,
        order=order,
        rank_offset=0,
    )

    # Build expert rank generator
    if expert_tensor_parallel_size is None:
        expert_tensor_parallel_size = tensor_model_parallel_size
    expert_tensor_model_pipeline_parallel_size = (
        expert_tensor_parallel_size * expert_model_parallel_size * pipeline_model_parallel_size
    )
    expert_data_parallel_size = decoder_world_size // expert_tensor_model_pipeline_parallel_size
    if decoder_world_size % expert_tensor_model_pipeline_parallel_size != 0:
        raise RuntimeError(
            f"decoder world_size ({decoder_world_size}) is not divisible by expert_tensor_model_pipeline_parallel size ({expert_tensor_model_pipeline_parallel_size})"
        )

    # TODO: support expert specific ordering
    expert_decoder_rank_generator = RankGenerator(
        tp=expert_tensor_parallel_size,
        ep=expert_model_parallel_size,
        dp=expert_data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=1,
        order=order,
        rank_offset=0,
    )

    assert (
        order.endswith("pp")
        or pipeline_model_parallel_size == 1
        or expert_data_parallel_size == data_parallel_size
    ), "When not using pp-last rank ordering, the data parallel size of the attention and moe layers must be the same"

    assert decoder_rank_generator.get_ranks("pp") == expert_decoder_rank_generator.get_ranks(
        "pp"
    ), f"Pipeline parallel groups are expected to be the same for Non-Expert and Expert part, \
    but got {decoder_rank_generator.get_ranks('pp')} and {expert_decoder_rank_generator.get_ranks('pp')}"

    def generator_wrapper(group_type, is_expert=False, **kwargs):
        """The `RankGenerator` class produces a hyper-rectangle for a given set of
        tensor, pipeline, data, expert, and context parallelism. If we have an encoder,
        in addition to the default decoder, we essentially instantiate two `RankGenerator`
        classes to construct the parallelism for each module separately, and we then have
        to stitch them together for the right groups. For now, this means pp and tp-pp.

        Let's say we have a total of 6 GPUs denoted by g0 ... g5.
        For encoder_tp=1, encoder_pp=1, decoder_tp=2, decoder_pp=1, dp=2,
        g0, g1 belong to encoder and g2, ..., g5 belong to decoder.
        The present function will create with "tp-dp-pp":
        3 data-parallel groups: [g0, g1], [g2, g4], [g3, g5]
        4 tensor model-parallel groups: [g0], [g1], [g2, g3], [g4, g5]
        4 pipeline model-parallel groups: [g0, g2], [g0, g3], [g1, g4], [g1, g5]
        """
        if is_expert:
            d_ranks = expert_decoder_rank_generator.get_ranks(group_type, **kwargs)
        else:
            d_ranks = decoder_rank_generator.get_ranks(group_type, **kwargs)

        if encoder_rank_generator is None:
            for x in d_ranks:
                yield x
            return

    # Build the data-parallel groups.
    _DATA_PARALLEL_GROUP = []
    _DATA_PARALLEL_GROUP_WITH_CP = []
    for ranks in generator_wrapper('dp'):
        _DATA_PARALLEL_GROUP.append(ranks)
    print(f"Data Parallel groups (DP): \n{_DATA_PARALLEL_GROUP}")

    for ranks_with_cp in generator_wrapper('dp-cp'):
        _DATA_PARALLEL_GROUP_WITH_CP.append(ranks_with_cp)
    print(f"Data Parallel groups with cp (DP_with_cp): \n{_DATA_PARALLEL_GROUP_WITH_CP}")

    # Build the context-parallel groups.
    _CONTEXT_PARALLEL_GROUP = []
    for ranks in generator_wrapper('cp'):
        _CONTEXT_PARALLEL_GROUP.append(ranks)
    print(f"Context Parallel groups (CP): \n{_CONTEXT_PARALLEL_GROUP}")

    # Build the model-parallel groups.
    _MODEL_PARALLEL_GROUP = []
    for ranks in generator_wrapper('tp-pp'):
        _MODEL_PARALLEL_GROUP.append(ranks)
    print(f"Model Parallel groups (MP): \n{_MODEL_PARALLEL_GROUP}")

    # Build the tensor model-parallel groups.
    _TENSOR_MODEL_PARALLEL_GROUP = []
    for ranks in generator_wrapper('tp'):
        _TENSOR_MODEL_PARALLEL_GROUP.append(ranks)
    print(f"Tensor Model Parallel groups (TP): \n{_TENSOR_MODEL_PARALLEL_GROUP}")
    
    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    _PIPELINE_MODEL_PARALLEL_GROUP = []
    for ranks in generator_wrapper('pp'):
        _PIPELINE_MODEL_PARALLEL_GROUP.append(ranks)
    print(f"Pipeline Model Parallel groups (PP): \n{_PIPELINE_MODEL_PARALLEL_GROUP}")

    # Build the tensor + data parallel groups.
    _TENSOR_AND_DATA_PARALLEL_GROUP = []
    _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = []
    for ranks in generator_wrapper('tp-dp-cp'):
        _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP.append(ranks)
    print(f"Tensor and Data Parallel groups with cp: \n{_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP}")

    for ranks in generator_wrapper('tp-dp'):
        _TENSOR_AND_DATA_PARALLEL_GROUP.append(ranks)
    print(f"Tensor and Data Parallel groups: \n{_TENSOR_AND_DATA_PARALLEL_GROUP}")

    _TENSOR_AND_CONTEXT_PARALLEL_GROUP = []
    for ranks in generator_wrapper('tp-cp'):
        _TENSOR_AND_CONTEXT_PARALLEL_GROUP.append(ranks)
    print(f"Tensor and Context Parallel groups (TCP): \n{_TENSOR_AND_CONTEXT_PARALLEL_GROUP}")

    ### Expert-related parallel groups initialization
    # Build the expert model parallel group
    _EXPERT_MODEL_PARALLEL_GROUP = []
    for ranks in generator_wrapper('ep', is_expert=True):
        _EXPERT_MODEL_PARALLEL_GROUP.append(ranks)
    print(f"Expert Model Parallel groups (EP): \n{_EXPERT_MODEL_PARALLEL_GROUP}")

    # Build the expert tensor parallel group
    _EXPERT_TENSOR_PARALLEL_GROUP = []
    for ranks in generator_wrapper('tp', is_expert=True):
        _EXPERT_TENSOR_PARALLEL_GROUP.append(ranks)
    print(f"Expert Tensor Parallel groups (ETP): \n{_EXPERT_TENSOR_PARALLEL_GROUP}")

    # Build the tensor + expert parallel groups
    _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = []
    for ranks in generator_wrapper('tp-ep', is_expert=True):
        _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP.append(ranks)
    print(f"Expert Tensor and Model Parallel groups: \n{_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP}")

    # Build the expert+tensor+pipeline parallel groups
    _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = []
    for ranks in generator_wrapper('tp-ep-pp', is_expert=True):
        _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP.append(ranks)
    print(f"Expert Tensor Model Pipeline Parallel groups: \n{_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP}")
    
    _EXPERT_DATA_PARALLEL_GROUP = []
    for ranks in generator_wrapper('dp', is_expert=True):
        _EXPERT_DATA_PARALLEL_GROUP.append(ranks)
    print(f"Expert Data Parallel groups (EDP): \n{_EXPERT_DATA_PARALLEL_GROUP}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=1, help="world size")
    parser.add_argument("--tp", type=int, default=1, help="tensor model parallel size")
    parser.add_argument("--pp", type=int, default=1, help="pipeline model parallel size")
    parser.add_argument("--ep", type=int, default=1, help="expert parallel size")
    parser.add_argument("--etp", type=int, default=None, help="expert tensor model parallel size")
    parser.add_argument("--cp", type=int, default=1, help="context parallel size")
    args = parser.parse_args()

    print("begin simulate model parallel group initialization...")
    print(f"world size: {args.world_size}, tp: {args.tp}, pp: {args.pp}, ep: {args.ep}, etp: {args.etp}, cp: {args.cp}")
    initialize_model_parallel(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        context_parallel_size=args.cp,
        expert_model_parallel_size=args.ep,
        expert_tensor_parallel_size=args.etp,
        world_size=args.world_size,
    )

#!!!!!!!!!!!!!!!!!!!!!
#这个代码也不是 demo 代码，跑不起来的，没有 main，没有任何启动
