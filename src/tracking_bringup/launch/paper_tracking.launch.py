import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    world_name_arg = DeclareLaunchArgument(
        "world_name", default_value="paper_sensors",
        description="bumperbot_description/gazebo.launch.py içindeki world_name arg"
    )

    layout_file_arg = DeclareLaunchArgument(
        "layout_file",
        default_value=os.path.join(os.path.expanduser("~"),
                                   "KalmanNet_Indoor_Tracking",
                                   "config",
                                   "paper_sensors_5x5_b20.csv"),
        description="Sensor layout CSV"
    )

    sigma_arg = DeclareLaunchArgument("sigma", default_value="0.10", description="Range noise sigma (m)")
    rate_arg  = DeclareLaunchArgument("rate",  default_value="10.0", description="Range publish rate (Hz)")
    delta_arg = DeclareLaunchArgument("delta", default_value="0.1",  description="EKF delta (s)")
    tau_arg   = DeclareLaunchArgument("tau",   default_value="1.0",  description="EKF tau")

    init_from_gt_arg = DeclareLaunchArgument(
        "init_from_gt", default_value="false",
        description="EKF init_from_gt (debug only) true/false"
    )

    tracking_ns_arg = DeclareLaunchArgument(
        "tracking_ns", default_value="tracking",
        description="Tracking pipeline namespace. '' yaparsan namespace kapatılır."
    )

    # ✅ NEW: sim-time switch
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
        description="Gazebo'da proxy model adı (gz_proxy_node bunu set_pose ile sürer)"
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

        sigma = LaunchConfiguration("sigma").perform(context)
        rate  = LaunchConfiguration("rate").perform(context)
        delta = LaunchConfiguration("delta").perform(context)
        tau   = LaunchConfiguration("tau").perform(context)
        init_from_gt = LaunchConfiguration("init_from_gt").perform(context).strip().lower()

        tracking_ns = LaunchConfiguration("tracking_ns").perform(context).strip().strip("/")
        ns_remap = f"__ns:=/{tracking_ns}" if tracking_ns else "__ns:=/"

        gz_world  = LaunchConfiguration("gz_world").perform(context)
        gz_entity = LaunchConfiguration("gz_entity").perform(context)

        # ✅ NEW: normalize use_sim_time to "true"/"false"
        use_sim_time_raw = LaunchConfiguration("use_sim_time").perform(context).strip().lower()
        use_sim_time = "true" if use_sim_time_raw in ("true", "1", "yes", "on") else "false"

        # --- Include: Gazebo ---
        gazebo_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory("bumperbot_description"),
                    "launch",
                    "gazebo.launch.py"
                )
            ),
            launch_arguments={"world_name": world_name}.items(),
        )

        # --- Include: Controller ---
        controller_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory("bumperbot_controller"),
                    "launch",
                    "controller.launch.py"
                )
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

        # --- Script paths ---
        ws_root = os.path.join(os.path.expanduser("~"), "KalmanNet_Indoor_Tracking")
        scripts_dir = os.path.join(ws_root, "scripts")

        gt_posearray_to_odom_py = os.path.join(scripts_dir, "gt_posearray_to_odom.py")
        range_measurements_py   = os.path.join(scripts_dir, "range_measurements.py")
        ekf_from_range_py       = os.path.join(scripts_dir, "ekf_tracker_from_range.py")
        metrics_py              = os.path.join(scripts_dir, "tracking_metrics_node.py")
        gz_proxy_py             = os.path.join(scripts_dir, "gz_proxy_node.py")
        viz_markers_py          = os.path.join(scripts_dir, "viz_tracking_markers.py")

        # Relative topic names (namespace altında toplanır)
        gt_odom_topic = "gt/odom"
        z_topic = "z"
        zmin_topic = "range/min"
        est_topic = "estimated"

        err_topic = "metrics/error"
        rmse_topic = "metrics/rmse"
        rmse_w_topic = "metrics/rmse_window"
        markers_topic = "viz/markers"

        # --- GT Odom node ---
        gt_odom = ExecuteProcess(
            cmd=[
                "python3", gt_posearray_to_odom_py,
                "--ros-args",
                "-r", ns_remap,
                "-r", "__node:=gt_posearray_to_odom",
                "-p", f"use_sim_time:={use_sim_time}",
                "-p", "pose_topic:=/gz/dynamic_poses",
                "-p", f"odom_topic:={gt_odom_topic}",
                "-p", "world_frame:=world",
                "-p", "child_frame:=base_link",
                "-p", "auto_pick:=true",
            ],
            output="screen",
        )

        # --- RANGE generator ---
        range_gen = ExecuteProcess(
            cmd=[
                "python3", range_measurements_py,
                "--ros-args",
                "-r", ns_remap,
                "-r", "__node:=range_measurement_generator",
                "-p", f"use_sim_time:={use_sim_time}",
                "-p", f"layout_file:={layout_file}",
                "-p", f"sigma:={sigma}",
                "-p", f"rate:={rate}",
                "-p", f"gt_topic:={gt_odom_topic}",
                "-p", f"z_topic:={z_topic}",
                "-p", f"min_topic:={zmin_topic}",
            ],
            output="screen",
        )

        # --- EKF (range) ---
        ekf = ExecuteProcess(
            cmd=[
                "python3", ekf_from_range_py,
                "--ros-args",
                "-r", ns_remap,
                "-r", "__node:=ekf",
                "-p", f"use_sim_time:={use_sim_time}",
                "-p", f"layout_file:={layout_file}",
                "-p", f"z_topic:={z_topic}",
                "-p", f"est_topic:={est_topic}",
                "-p", f"sigma:={sigma}",
                "-p", f"delta:={delta}",
                "-p", f"tau:={tau}",
                "-p", f"init_from_gt:={init_from_gt}",
                "-p", f"gt_topic:={gt_odom_topic}",
            ],
            output="screen",
        )

        # --- Metrics node ---
        metrics = ExecuteProcess(
            cmd=[
                "python3", metrics_py,
                "--ros-args",
                "-r", ns_remap,
                "-r", "__node:=metrics",
                "-p", f"use_sim_time:={use_sim_time}",
                "-p", f"gt_topic:={gt_odom_topic}",
                "-p", f"est_topic:={est_topic}",
                "-p", f"error_topic:={err_topic}",
                "-p", f"rmse_topic:={rmse_topic}",
                "-p", f"rmse_window_topic:={rmse_w_topic}",
                "-p", "rmse_window_N:=200",
            ],
            output="screen",
        )

        # --- Gazebo proxy node ---
        gz_proxy = ExecuteProcess(
            cmd=[
                "python3", gz_proxy_py,
                "--ros-args",
                "-r", ns_remap,
                "-r", "__node:=gz_proxy",
                "-p", f"use_sim_time:={use_sim_time}",
                "-p", f"est_topic:={est_topic}",
                "-p", f"gz_world:={gz_world}",
                "-p", f"gz_entity:={gz_entity}",
                "-p", "gz_z:=0.01",
            ],
            output="screen",
        )

        # --- Marker viz node ---
        viz = ExecuteProcess(
            cmd=[
                "python3", viz_markers_py,
                "--ros-args",
                "-r", ns_remap,
                "-r", "__node:=viz",
                "-p", f"use_sim_time:={use_sim_time}",
                "-p", f"layout_file:={layout_file}",
                "-p", "world_frame:=world",
                "-p", f"gt_topic:={gt_odom_topic}",
                "-p", f"est_topic:={est_topic}",
                "-p", f"marker_topic:={markers_topic}",
            ],
            output="screen",
        )

        # --- world->odom TF (opsiyonel) ---
        static_tf = Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_world_to_odom",
            output="screen",
            arguments=["0", "0", "0", "0", "0", "0", "world", "odom"],
            condition=IfCondition(LaunchConfiguration("publish_world_tf")),
        )

        # --- RViz2 (opsiyonel) ---
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

        # Gazebo başladıktan sonra pipeline’ı başlat
        pipeline_delayed = TimerAction(
            period=2.0,
            actions=[bridge_dynamicposes, gt_odom, range_gen, ekf, metrics, gz_proxy, viz, static_tf, rviz],
        )

        return [gazebo_launch, controller_delayed, pipeline_delayed]

    return LaunchDescription([
        world_name_arg,
        layout_file_arg,
        sigma_arg, rate_arg, delta_arg, tau_arg,
        init_from_gt_arg,
        tracking_ns_arg,
        use_sim_time_arg,          # ✅ added
        gz_world_arg, gz_entity_arg,
        publish_world_tf_arg,
        use_rviz_arg, rviz_config_arg,
        OpaqueFunction(function=launch_setup),
    ])
