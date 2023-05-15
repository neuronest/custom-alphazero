import os

tensorflow_log_level = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = tensorflow_log_level


class ConfigGeneral:
    game = "connect_n"
    # -1 is used to disable the GPU
    # other values should be used only once
    self_play_gpu_index = "-1"
    serving_gpu_index = "-1"
    training_gpu_index = "0"
    concurrency = False
    mono_process = False
    http_inference = False


class ConfigSelfPlay:
    discounting_factor = 1  # set to 1 to actually disable any discounting effect
    samples_checkpoint_frequency = 1
    mcts_iterations = 250
    exclude_null_games = True


class ConfigChess:
    piece_symbols = [None, "p", "n", "b", "r", "q", "k"]
    initial_board_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    initial_turn = "w"
    initial_castling_rights = "KQkq"
    initial_ep_quare = "-"
    initial_halfmove_clock = "0"
    initial_fullmove_number = "1"
    board_size = 8
    number_unique_pieces = 12


class ConfigConnectN:
    board_width = 7
    board_height = 6
    n = 4
    gravity = True
    black = -1
    empty = 0
    white = 1
    pieces = {-1: "O", 0: ".", 1: "X"}
    directions = [(0, 1), (1, 1), (1, 0), (1, -1)]


class ConfigMCTS:
    exploration_constant = 1.5
    enable_dirichlet_noise = False  # disabled for now
    dirichlet_noise_value = 0.03
    dirichlet_noise_ratio = 0.25
    index_move_greedy = 8  # 30 should be the default value
    use_solver = False


class ConfigModel:
    training_epochs = 1
    batch_size = 256
    l2_penalization_term = 1e-4
    depth = 4
    maximum_learning_rate = 1e-2
    learning_rates = {
        range(0, 150000): 1e-2,
        range(150000, 300000): 1e-3,
    }
    minimum_learning_rate = 1e-4
    momentum = 0.9
    filters = 128


class ConfigServing:
    serving_host = "localhost"
    serving_port = 5555
    serving_address = "http://{0}:{1}".format(serving_host, serving_port)
    """
    import multiprocessing
    # seems not to run as fast as expected for now
    inference_batch_size = multiprocessing.cpu_count() - 1
    """
    minimum_training_size = 2500
    samples_queue_size = 10000
    inference_batch_size = 1
    inference_timeout = 1
    model_evaluation_frequency = 50
    model_checkpoint_frequency = 50
    evaluation_games_number = 150
    replace_min_score = 0.55
    training_loop_sleep_time = 0.5
    evaluate_with_mcts = False
    evaluate_with_solver = False


class ConfigPath:
    # serving APIs
    run_id_path = "/api/run-id"  # GET endpath
    queue_path = "/api/queue"
    append_queue_path = queue_path + "/append"  # PATCH endpath
    retrieve_queue_path = queue_path + "/retrieve"  # PUT endpath
    size_queue_path = queue_path + "/size"  # GET endpath
    best_model_path = "/api/best-model"
    update_best_model_path = best_model_path + "/update"  # PUT endpath
    inference_path = "/api/inference"  # POST endpath
    # disk paths
    results_dir = "results"  # results/
    self_play_dir = "self_play"  # results/{game}/{run_id}/self_play
    training_dir = "training"  # results/{game}/{run_id}/training
    evaluation_dir = "evaluation"  # results/{game}/{run_id}/evaluation
    samples_file = (
        "samples.npz"  # results/{game}/{run_id}/self_play/{iteration}/samples.npz
    )
    """
    results/{game}/{run_id}/training/model*
    results/{game}/{run_id}/evaluation/{iteration}/model*
    """
    model_prefix = "model"
    model_meta = "meta.json"
    model_success = "MODEL_SAVED_SUCCESSFULLY"
    updated_mcts_dir = "updated_mcts"  # results/{game}/{run_id}/self_play/updated_mcts
    tensorboard_dir = "tensorboard"  # results/{game}/{run_id}/tensorboard
    # exact solver
    connect4_solver_bin = "./src/exact_solvers/c4solver"
    connect4_opening_book = "./src/exact_solvers/7x6.book"
