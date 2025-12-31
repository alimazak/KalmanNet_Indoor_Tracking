import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ---------------- Launch args ----------------
    world_name_arg = DeclareLaunchArgument(
        "world_name",
        default_value="paper_sensors",
        description="bumperbot_description/gazebo.launch.py içindeki world_name arg (world file seçimi)",
    )

    layout_file_arg = DeclareLaunchArgument(
        "layout_file",
        default_value=os.path.join(
            os.path.expanduser("~"),
            "KalmanNet_Indoor_Tracking",
            "config",
            "paper_sensors_5x5_b20.csv",
        ),
        description="Sensor layout CSV",
    )

    sigma_arg = DeclareLaunchArgument("sigma", default_value="0.10", description="Range noise sigma (m)")
    rate_arg = DeclareLaunchArgument("rate", default_value="10.0", description="Range publish rate (Hz)")
    delta_arg = DeclareLaunchArgument("delta", default_value="0.1", description="EKF delta (s)")
    tau_arg = DeclareLaunchArgument("tau", default_value="1.0", description="EKF tau")

    # IMPORTANT: default FALSE (GT init kapalı)
    init_from_gt_arg = DeclareLaunchArgument(
        "init_from_gt",
        default_value="false",
        description="EKF init_from_gt (true/false). Default: false",
    )

    gz_world_arg = DeclareLaunchArgument(
        "gz_world",
        default_value="empty_world",
        description="Gazebo world name (SetEntityPose + dynamic_pose paths). Servisinde genelde empty_world.",
    )
    gz_entity_arg = DeclareLaunchArgument(
        "gz_entity",
        default_value="ekf_proxy",
        description="Gazebo'da EKF proxy model adı (ekf_tracker_from_range.py set_pose ile sürüyor)",
    )

    publish_world_tf_arg = DeclareLaunchArgument(
        "publish_world_tf",
        default_value="true",
        description="world->odom static TF yayınla (RViz boşsa işe yarar).",
    )

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="false",
        description="RViz2 otomatik açılsın mı?",
    )
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value="",
        description="Opsiyonel RViz config dosyası (boş bırakabilirsin)",
    )

    def launch_setup(context, *args, **kwargs):
        world_name = LaunchConfiguration("world_name").perform(context)
        layout_file = LaunchConfiguration("layout_file").perform(context)

        sigma = LaunchConfiguration("sigma").perform(context)
        rate = LaunchConfiguration("rate").perform(context)
        delta = LaunchConfiguration("delta").perform(context)
        tau = LaunchConfiguration("tau").perform(context)
        init_from_gt = LaunchConfiguration("init_from_gt").perform(context)

        gz_world = LaunchConfiguration("gz_world").perform(context)
        gz_entity = LaunchConfiguration("gz_entity").perform(context)

        # ---------------- Include: Gazebo ----------------
        gazebo_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory("bumperbot_description"),
                    "launch",
                    "gazebo.launch.py",
                )
            ),
            launch_arguments={"world_name": world_name}.items(),
        )

        # ---------------- Include: Controller ----------------
        controller_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory("bumperbot_controller"),
                    "launch",
                    "controller.launch.py",
                )
            )
        )
        controller_delayed = TimerAction(period=3.0, actions=[controller_launch])

        # ---------------- Bridge: Gazebo -> ROS PoseArray ----------------
        # (Node name collision fix): name=bridge_dynamicposes
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

        # ---------------- Script paths ----------------
        ws_root = os.path.join(os.path.expanduser("~"), "KalmanNet_Indoor_Tracking")
        scripts_dir = os.path.join(ws_root, "scripts")

        gt_posearray_to_odom_py = os.path.join(scripts_dir, "gt_posearray_to_odom.py")
        range_measurements_py = os.path.join(scripts_dir, "range_measurements.py")
        ekf_from_range_py = os.path.join(scripts_dir, "ekf_tracker_from_range.py")
        viz_markers_py = os.path.join(scripts_dir, "viz_tracking_markers.py")

        # ---------------- Ground Truth Odom ----------------
        gt_odom = ExecuteProcess(
            cmd=[
                "python3",
                gt_posearray_to_odom_py,
                "--ros-args",
                "-p",
                "pose_topic:=/gz/dynamic_poses",
                "-p",
                "odom_topic:=/ground_truth/odom",
                "-p",
                "world_frame:=world",
                "-p",
                "child_frame:=base_link",
                "-p",
                "auto_pick:=true",
            ],
            output="screen",
        )

        # ---------------- RANGE measurement generator ----------------
        range_gen = ExecuteProcess(
            cmd=[
                "python3",
                range_measurements_py,
                "--ros-args",
                "-p",
                f"layout_file:={layout_file}",
                "-p",
                f"sigma:={sigma}",
                "-p",
                f"rate:={rate}",
                "-p",
                "gt_topic:=/ground_truth/odom",
                "-p",
                "z_topic:=/range/z",
            ],
            output="screen",
        )

        # ---------------- EKF (range) ----------------
        ekf = ExecuteProcess(
            cmd=[
                "python3",
                ekf_from_range_py,
                "--ros-args",
                "-p",
                f"layout_file:={layout_file}",
                "-p",
                "z_topic:=/range/z",
                "-p",
                f"sigma:={sigma}",
                "-p",
                f"delta:={delta}",
                "-p",
                f"tau:={tau}",
                # IMPORTANT: init_from_gt launch arg (default false)
                "-p",
                f"init_from_gt:={init_from_gt}",
                "-p",
                f"gz_world:={gz_world}",
                "-p",
                f"gz_entity:={gz_entity}",
            ],
            output="screen",
        )

        # ---------------- Marker viz ----------------
        viz = ExecuteProcess(
            cmd=[
                "python3",
                viz_markers_py,
                "--ros-args",
                "-p",
                f"layout_file:={layout_file}",
                "-p",
                "world_frame:=world",
            ],
            output="screen",
        )

        # ---------------- world->odom TF (optional) ----------------
        static_tf = Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_world_to_odom",
            output="screen",
            arguments=["0", "0", "0", "0", "0", "0", "world", "odom"],
            condition=IfCondition(LaunchConfiguration("publish_world_tf")),
        )

        # ---------------- RViz2 (optional) ----------------
        rviz_cfg = LaunchConfiguration("rviz_config").perform(context)
        if rviz_cfg.strip():
            rviz = Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2_tracking",
                output="screen",
                arguments=["-d", rviz_cfg],
                condition=IfCondition(LaunchConfiguration("use_rviz")),
            )
        else:
            rviz = Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2_tracking",
                output="screen",
                condition=IfCondition(LaunchConfiguration("use_rviz")),
            )

        # Gazebo biraz ayağa kalksın diye pipeline'ı geciktiriyoruz
        pipeline_delayed = TimerAction(
            period=2.0,
            actions=[bridge_dynamicposes, gt_odom, range_gen, ekf, viz, static_tf, rviz],
        )

        return [gazebo_launch, controller_delayed, pipeline_delayed]

    return LaunchDescription(
        [
            world_name_arg,
            layout_file_arg,
            sigma_arg,
            rate_arg,
            delta_arg,
            tau_arg,
            init_from_gt_arg,
            gz_world_arg,
            gz_entity_arg,
            publish_world_tf_arg,
            use_rviz_arg,
            rviz_config_arg,
            OpaqueFunction(function=launch_setup),
        ]
    )