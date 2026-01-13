import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def _find_ws_root() -> Path:
    """
    Find workspace/repo root by walking up parents until we see:
      - scripts/
      - src/
    Works both from source tree and from install/ share/ path.
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "scripts").is_dir() and (p / "src").is_dir():
            return p
    return Path.home() / "KalmanNet_Indoor_Tracking"


def generate_launch_description():
    ws_root = _find_ws_root()
    default_layout = str(ws_root / "config" / "paper_sensors_5x5_b20.csv")

    # ---------------- Launch args ----------------
    world_name_arg = DeclareLaunchArgument(
        "world_name", default_value="paper_sensors",
        description="bumperbot_description/gazebo.launch.py içindeki world_name arg"
    )

    layout_file_arg = DeclareLaunchArgument(
        "layout_file",
        default_value=default_layout,
        description="Sensor layout CSV"
    )

    sigma_arg = DeclareLaunchArgument("sigma", default_value="0.10", description="Range noise sigma (m)")
    sigma_meas_arg = DeclareLaunchArgument(
        "sigma_meas", default_value="",
        description="True measurement noise used by range generator. Empty => use sigma"
    )
    sigma_ekf_arg = DeclareLaunchArgument(
        "sigma_ekf", default_value="",
        description="EKF assumed measurement noise (R). Empty => use sigma"
    )

    rate_arg = DeclareLaunchArgument("rate", default_value="10.0", description="Range publish rate (Hz)")
    delta_arg = DeclareLaunchArgument("delta", default_value="0.1", description="EKF delta (s)")
    tau_arg = DeclareLaunchArgument("tau", default_value="1.0", description="EKF tau")

    init_from_gt_arg = DeclareLaunchArgument(
        "init_from_gt", default_value="false",
        description="EKF init_from_gt (debug only) true/false"
    )

    tracking_ns_arg = DeclareLaunchArgument(
        "tracking_ns", default_value="tracking",
        description="Tracking pipeline namespace. '' yaparsan namespace root olur."
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="true",
        description="Use Gazebo /clock (sim time) for tracking pipeline nodes."
    )

    gz_world_arg = DeclareLaunchArgument(
        "gz_world", default_value="empty_world",
        description="Gazebo world name (SetEntityPose service yolu için)."
    )

    gz_entity_arg = DeclareLaunchArgument(
        "gz_entity", default_value="ekf_proxy",
        description="Gazebo'da EKF proxy model adı (gz_proxy_node bunu set_pose ile sürer)"
    )

    # --- NEW: KNet proxy args ---
    enable_knet_proxy_arg = DeclareLaunchArgument(
        "enable_knet_proxy", default_value="false",
        description="Gazebo'da ikinci proxy (KNet) gösterilsin mi?"
    )
    knet_est_topic_arg = DeclareLaunchArgument(
        "knet_est_topic", default_value="knet/estimated",
        description="KNet estimated odom topic (tracking namespace altında, relative)."
    )
    gz_entity_knet_arg = DeclareLaunchArgument(
        "gz_entity_knet", default_value="knet_proxy",
        description="Gazebo'da KNet proxy model adı"
    )
    gz_z_knet_arg = DeclareLaunchArgument(
        "gz_z_knet", default_value="0.02",
        description="KNet proxy için Gazebo z offset"
    )

    publish_world_tf_arg = DeclareLaunchArgument(
        "publish_world_tf", default_value="true",
        description="world->odom static TF yayınla"
    )

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz", default_value="false",
        description="RViz2 otomatik açılsın mı?"
    )
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config", default_value="",
        description="Opsiyonel RViz config dosyası"
    )

    def launch_setup(context, *args, **kwargs):
        world_name = LaunchConfiguration("world_name").perform(context)
        layout_file = LaunchConfiguration("layout_file").perform(context)

        sigma = LaunchConfiguration("sigma").perform(context).strip()
        sigma_meas = LaunchConfiguration("sigma_meas").perform(context).strip()
        sigma_ekf = LaunchConfiguration("sigma_ekf").perform(context).strip()

        # IMPORTANT: Empty => fallback to sigma (prevents range/ekf scripts crashing)
        if not sigma_meas:
            sigma_meas = sigma
        if not sigma_ekf:
            sigma_ekf = sigma

        rate = LaunchConfiguration("rate").perform(context)
        delta = LaunchConfiguration("delta").perform(context)
        tau = LaunchConfiguration("tau").perform(context)
        init_from_gt = LaunchConfiguration("init_from_gt").perform(context).strip().lower()

        tracking_ns = LaunchConfiguration("tracking_ns").perform(context).strip().strip("/")
        ns_remap = f"__ns:=/{tracking_ns}" if tracking_ns else "__ns:=/"

        use_sim_time = LaunchConfiguration("use_sim_time").perform(context).strip().lower()
        use_sim_time = "true" if use_sim_time in ("true", "1", "yes", "on") else "false"

        gz_world = LaunchConfiguration("gz_world").perform(context)
        gz_entity = LaunchConfiguration("gz_entity").perform(context)

        # KNet proxy params
        knet_est_topic = LaunchConfiguration("knet_est_topic").perform(context)
        gz_entity_knet = LaunchConfiguration("gz_entity_knet").perform(context)
        gz_z_knet = LaunchConfiguration("gz_z_knet").perform(context)

        # --- Gazebo include ---
        gazebo_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory("bumperbot_description"), "launch", "gazebo.launch.py")
            ),
            launch_arguments={"world_name": world_name}.items(),
        )

        # --- Controller include (delayed) ---
        controller_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory("bumperbot_controller"), "launch", "controller.launch.py")
            )
        )
        controller_delayed = TimerAction(period=3.0, actions=[controller_launch])

        # --- Bridge: Gazebo -> ROS PoseArray ---
        bridge_topic = f"/world/{gz_world}/dynamic_pose/info@geometry_msgs/msg/PoseArray[gz.msgs.Pose_V"
        bridge_ros_in = f"/world/{gz_world}/dynamic_pose/info"

        bridge_dynamicposes = Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="bridge_dynamicposes",
            output="screen",
            arguments=[bridge_topic],
            remappings=[(bridge_ros_in, "/gz/dynamic_poses")],
        )

        # --- Scripts dir (auto-detected) ---
        scripts_dir = ws_root / "scripts"

        def py_exec(script_name: str, node_name: str, params: dict, condition=None) -> ExecuteProcess:
            script_path = str(scripts_dir / script_name)
            cmd = [
                "python3",
                script_path,
                "--ros-args",
                "-r", ns_remap,
                "-r", f"__node:={node_name}",
                "-p", f"use_sim_time:={use_sim_time}",
            ]
            for k, v in params.items():
                cmd += ["-p", f"{k}:={v}"]
            return ExecuteProcess(cmd=cmd, output="screen", condition=condition)

        # Relative topic names (namespace altında)
        gt_odom_topic = "gt/odom"
        z_topic = "z"
        zmin_topic = "range/min"
        est_topic = "estimated"

        err_topic = "metrics/error"
        rmse_topic = "metrics/rmse"
        rmse_w_topic = "metrics/rmse_window"
        markers_topic = "viz/markers"

        # --- Nodes (python scripts) ---
        gt_odom = py_exec(
            "gt_posearray_to_odom.py",
            "gt_posearray_to_odom",
            {
                "pose_topic": "/gz/dynamic_poses",
                "odom_topic": gt_odom_topic,
                "world_frame": "world",
                "child_frame": "base_link",
                "auto_pick": "true",
            },
        )

        range_gen = py_exec(
            "range_measurements.py",
            "range_measurement_generator",
            {
                "layout_file": layout_file,
                "sigma": sigma_meas,
                "rate": rate,
                "gt_topic": gt_odom_topic,
                "z_topic": z_topic,
                "min_topic": zmin_topic,
            },
        )

        ekf = py_exec(
            "ekf_tracker_from_range.py",
            "ekf",
            {
                "layout_file": layout_file,
                "z_topic": z_topic,
                "est_topic": est_topic,
                "sigma": sigma_ekf,
                "delta": delta,
                "tau": tau,
                "init_from_gt": init_from_gt,
                "gt_topic": gt_odom_topic,
            },
        )

        metrics = py_exec(
            "tracking_metrics_node.py",
            "metrics",
            {
                "gt_topic": gt_odom_topic,
                "est_topic": est_topic,
                "error_topic": err_topic,
                "rmse_topic": rmse_topic,
                "rmse_window_topic": rmse_w_topic,
                "rmse_window_N": "200",
            },
        )

        # EKF proxy (Gazebo)
        gz_proxy_ekf = py_exec(
            "gz_proxy_node.py",
            "gz_proxy",
            {
                "est_topic": est_topic,
                "gz_world": gz_world,
                "gz_entity": gz_entity,
                "gz_z": "0.01",
                "rate_hz": rate,
            },
        )

        # NEW: KNet proxy (Gazebo)
        gz_proxy_knet = py_exec(
            "gz_proxy_node.py",
            "gz_proxy_knet",
            {
                "est_topic": knet_est_topic,   # e.g. "knet/estimated"
                "gz_world": gz_world,
                "gz_entity": gz_entity_knet,   # e.g. "knet_proxy"
                "gz_z": gz_z_knet,
                "rate_hz": rate,
            },
            condition=IfCondition(LaunchConfiguration("enable_knet_proxy")),
        )

        viz = py_exec(
            "viz_tracking_markers.py",
            "viz",
            {
                "layout_file": layout_file,
                "world_frame": "world",
                "gt_topic": gt_odom_topic,
                "est_topic": est_topic,
                "marker_topic": markers_topic,
            },
        )

        static_tf = Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_world_to_odom",
            output="screen",
            arguments=["0", "0", "0", "0", "0", "0", "world", "odom"],
            condition=IfCondition(LaunchConfiguration("publish_world_tf")),
        )

        # --- RViz2 (optional) ---
        rviz_cfg = LaunchConfiguration("rviz_config").perform(context)
        if rviz_cfg.strip():
            rviz = Node(
                package="rviz2",
                executable="rviz2",
                output="screen",
                arguments=["-d", rviz_cfg],
                condition=IfCondition(LaunchConfiguration("use_rviz")),
                parameters=[{"use_sim_time": (use_sim_time == "true")}],
            )
        else:
            rviz = Node(
                package="rviz2",
                executable="rviz2",
                output="screen",
                condition=IfCondition(LaunchConfiguration("use_rviz")),
                parameters=[{"use_sim_time": (use_sim_time == "true")}],
            )

        pipeline_delayed = TimerAction(
            period=2.0,
            actions=[
                bridge_dynamicposes,
                gt_odom,
                range_gen,
                ekf,
                metrics,
                gz_proxy_ekf,
                gz_proxy_knet,   # <- KNet proxy burada
                viz,
                static_tf,
                rviz,
            ],
        )

        return [gazebo_launch, controller_delayed, pipeline_delayed]

    return LaunchDescription([
        world_name_arg,
        layout_file_arg,
        sigma_arg, sigma_meas_arg, sigma_ekf_arg, rate_arg, delta_arg, tau_arg,
        init_from_gt_arg,
        tracking_ns_arg,
        use_sim_time_arg,
        gz_world_arg, gz_entity_arg,

        # NEW args
        enable_knet_proxy_arg, knet_est_topic_arg, gz_entity_knet_arg, gz_z_knet_arg,

        publish_world_tf_arg,
        use_rviz_arg, rviz_config_arg,
        OpaqueFunction(function=launch_setup),
    ])
