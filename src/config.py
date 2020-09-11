import os

from ray.tune import grid_search, sample_from
import numpy as np

gpu_index = "-1"
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
    maximum_learning_rate = 1e-2
    learning_rates = {
        range(0, 150000): 1e-2,
        range(150000, 300000): 1e-3,
    }
    minimum_learning_rate = 1e-4
    momentum = 0.9
    filters = 128
    strides = (1, 1)
    padding = "same"
    activation = "relu"
    batch_normalization = True
    residual_residual_connexion = True
    residual_depth = 4
    value_hidden_dim = 256
    value_kernel_size = 1
    residual_kernel_size = 2
    policy_kernel_size = 1


class ConfigServing:
    serving_host = "localhost"
    serving_port = 5000
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
    archi_searches_dir = (
        "architecture_search"  # results/{game}/{run_id}/architecture_search
    )
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


class ConfigArchiSearch:
    # run_id is on the %Y-%m-%d-%H%M%S format
    # must be the run_id of a past run of self-play and training
    run_id = ""
    iteration = 28
    report_filename = "search_report"
    # if "debug" then it will run on a single process and be easier to debug
    running_mode = ""
    search_mode = "min"
    search_stop_criterion = {
        # "mean_accuracy": 0.80,
        "training_iteration": 6,
    }
    resources_per_trial = {"cpu": 4, "gpu": 1}
    perturbation_interval = 2
    num_samples_from_config = 20
    hyperparam_config = {
        "l2_penalization_term": sample_from(
            lambda _: np.random.choice(np.linspace(1e-4 / 10, 1e-4 * 10, 5))
        ),
        "maximum_learning_rate": sample_from(
            lambda _: np.random.choice(np.linspace(1e-2 / 10, 1e-2 * 10, 5))
        ),
        "momentum": sample_from(
            lambda _: np.random.choice(np.linspace(0.9 / 10, 0.95, 5))
        ),
        "filters": sample_from(lambda _: np.random.choice([32, 64, 128, 256])),
        "strides": (1, 1),
        "padding": "same",
        "activation": "relu",
        "batch_normalization": sample_from(lambda _: np.random.choice([True, False])),
        "residual_residual_connexion": sample_from(
            lambda _: np.random.choice([True, False])
        ),
        "residual_depth": sample_from(lambda _: np.random.choice([2, 4, 8, 16])),
        "value_hidden_dim": sample_from(
            lambda _: np.random.choice([64, 128, 256, 512])
        ),
        # gridsearch with several values increases a lot the number of trials
        "value_kernel_size": grid_search([1, 2, 3]),
        "residual_kernel_size": grid_search([1, 2, 3]),
        "policy_kernel_size": grid_search([1, 2, 3]),
    }
    hyperparam_mutations = {
        # list instead of lambda that throws errors sometimes
        # must send a key that matches an argument in the model constructor
        "maximum_learning_rate": np.linspace(0, 1, 20).tolist(),
        "momentum": np.linspace(0, 1, 20).tolist(),
    }

    # analysis rendering section
    discrete_hyperparameters = [
        "activation",
        "batch_normalization",
        "filters",
        "padding",
        "policy_kernel_size",
        "residual_depth",
        "residual_kernel_size",
        "residual_residual_connexion",
        "strides",
        "value_hidden_dim",
        "value_kernel_size",
    ]
    continuous_hyperparameters = [
        "l2_penalization_term",
        "maximum_learning_rate",
        "momentum",
    ]
    # architecture search report is a dictonary
    # dictionary keys and the types of information stored are defined here
    report_config = (
        "best_config"  # dict mapps hyperparameter name to their optimal value
    )
    reports_trials = "trials"  # several DataFrame
    report_on_trials = "aggregated_trials"  # all trials information in one DataFrame
    # sums up information about the mean loss per hyperparameter value over trials
    report_hyperparameters_contribution = "hyperparameters_contribution"
