import os
import time
import mujoco_py
import copy
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import MujocoEnv
from scipy.spatial.transform import Rotation

DEFAULT_CAMERA_CONFIG = {
    'azimuth': -158.83902439024388,
    'distance': 6.810744473960585,
    'elevation': -15.321951219512261,
    'fixedcamid': -1,
    'lookat': np.array([-7.25409872e-12, -3.33514935e-01, 1.36040619e+00])
}


class ContextualBallCatching(MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='model.xml',
                 reset_noise_scale=0.05,
                 context=np.array([0.68, 0.9, 0.85])):
        utils.EzPickle.__init__(**locals())

        self.p_gains = np.array([200, 300, 100, 100, 10 * 2, 10 * 2, 2.5 * 2])
        self.d_gains = np.array([7, 15, 5, 2.5, 0.3 * 1.41, 0.3 * 1.41, 0.05 * 1.41])

        self._reset_noise_scale = reset_noise_scale
        self.context = context

        xml_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "barrett")
        xml_path = os.path.join(xml_dir, xml_file)
        self.step = self.dummy_step

        # We fix the first and third joint so that the robot only moves in the plane
        self.des_pos = np.zeros(7)
        self._target_pos = None

        MujocoEnv.__init__(self, xml_path, 5)

        self.init_qpos = np.array([0., 0.4, 0., 1.5, 0., 0., 0.])
        self.init_qvel = np.array([0., 0., 0., 0., 0., 0., 0.])

        self.step = self.real_step
        self.ball_body_id = self.sim.model._body_name2id["ball"]
        self.net_body_id = self.sim.model._body_name2id["wam/net"]

        self.ball_geom_id = self.sim.model._geom_name2id["ball_geom"]
        self.net_geom_id = self.sim.model._geom_name2id["net_geom"]
        self.ground_geom_id = self.sim.model._geom_name2id["ground"]

    def _set_action_space(self):
        scale_array = 2. * np.array([1.985, 2.02, 2.9, 1.57, 2.7])
        self.action_space = spaces.Box(low=-scale_array, high=scale_array, dtype=np.float32)
        return self.action_space

    def control_cost(self, action):
        # Maximum resulting control penalty is 0.5
        control_cost = 5e-3 * np.sum(np.square(action))
        return control_cost

    def update_target_distribution_visualization(self, mu_target, sigma_target):
        target_body_id = self.sim.model._body_name2id["target_dist"]
        target_geom_id = self.sim.model._geom_name2id["target_dist_geom"]

        w, v = np.linalg.eig(sigma_target)
        e = np.eye(2)
        r = np.eye(3)
        for i in range(0, 2):
            for j in range(0, 2):
                r[i, j] = np.dot(v[:, i], e[:, j])

        sizes = 2.2 * np.sqrt(w)
        quat = Rotation.from_dcm(r).as_quat()

        self.sim.model.body_pos[target_body_id][1:] = mu_target
        self.sim.data.body_xpos[target_body_id][1:] = mu_target

        self.sim.model.geom_rbound[target_geom_id] = np.max(sizes)
        self.sim.model.geom_size[target_geom_id][1:] = sizes

        self.sim.model.body_quat[target_body_id][:] = quat
        self.sim.data.body_xquat[target_body_id][:] = quat
        self.sim.data.body_xmat[target_body_id][:] = r.reshape((-1,))

    def dummy_step(self, action):
        observation = self._get_obs()
        return observation, 0., False, {}

    def real_step(self, action):
        net_touched = False
        ground_touched = False

        # The action is the displacement and we assume to have a desired velocity of 0 in all joints
        for _ in range(self.frame_skip):
            self._joint_position_control(action)
            self.sim.step()

            # Check whether we caught the ball
            net_touched, ground_touched = self.get_collisions()
            if net_touched or ground_touched:
                break

        control_cost = self.control_cost(action)

        if net_touched:
            net_normal = np.reshape(self.sim.data.body_xmat[self.net_body_id, :], (3, 3))[2, :]
            norm_ball_vel = self.sim.data.qvel[7:10] / np.linalg.norm(self.sim.data.qvel[7:10])
            catch_reward = 50. + 25 * (np.dot(net_normal, norm_ball_vel) ** 5)
        else:
            catch_reward = 0.

        observation = self._get_obs()
        return observation, 0.275 - control_cost + catch_reward, net_touched or ground_touched, {"success": net_touched}

    def _joint_position_control(self, actions):
        self.des_pos[1] += self.model.opt.timestep * actions[0]
        self.des_pos[3:] += self.model.opt.timestep * actions[1:]
        torques = self.p_gains * (self.des_pos - self.sim.data.qpos[0:7]) - self.d_gains * self.sim.data.qvel[0:7]
        self.sim.data.ctrl[:] = torques

    def _get_obs(self):
        position = np.concatenate(([self.sim.data.qpos[1]], self.sim.data.qpos[3:].flat.copy()))
        velocity = np.concatenate(([self.sim.data.qvel[1]], self.sim.data.qvel[3:].flat.copy()))
        desired_position = np.concatenate(([self.des_pos[1]], self.des_pos[3:].copy()))
        observations = np.concatenate((desired_position, position, velocity))
        observations = np.clip(observations, -10., 10.)

        return observations

    def get_collisions(self):
        for coni in range(0, self.sim.data.ncon):
            con = self.sim.data.contact[coni]

            collision1 = con.geom1 == self.ball_geom_id and con.geom2 == self.net_geom_id
            collision2 = con.geom1 == self.net_geom_id and con.geom2 == self.ball_geom_id

            # This means we have a collision with one of the obstacles
            if collision1 or collision2:
                return True, False
            # Anything else means that we stop the simulation
            elif con.geom1 == self.ball_geom_id or con.geom2 == self.ball_geom_id:
                return False, True

        return False, False

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        # Sample a random starting position for the ball
        init_yz_limits = np.array([[-0.75, -0.65], [0.8, 1.8]])
        init_yz_ball_pos = np.random.uniform(init_yz_limits[:, 0], init_yz_limits[:, 1])

        # Now we compute the required initial velocity to reach the target position at this point
        target_ball_pos = np.array([0.,
                                    -np.cos(self.context[0]) * self.context[1],
                                    0.75 + np.sin(self.context[0]) * self.context[1]])
        self._target_pos = np.copy(target_ball_pos)
        target_body = self.sim.model._body_name2id["target"]
        self.sim.model.body_pos[target_body][:] = target_ball_pos
        self.sim.data.body_xpos[target_body][:] = target_ball_pos
        init_ball_pos = np.concatenate(([self.context[2]], init_yz_ball_pos))

        # Target is given, so draw a random time interval and compute the initial velocity
        t = 0.5 + 0.05 * self.context[2]

        init_ball_vel = (target_ball_pos - init_ball_pos + 0.5 * 9.81 * np.array([0., 0., 1.]) * (t ** 2)) / t

        pos_noise = self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq - 3)
        pos_noise[[0, 2]] = 0.
        vel_noise = self._reset_noise_scale * self.np_random.randn(self.model.nv - 3)
        vel_noise[[0, 2]] = 0.
        qpos = np.concatenate((self.init_qpos + pos_noise, init_ball_pos))
        qvel = np.concatenate((self.init_qvel + vel_noise, init_ball_vel))

        self.set_state(qpos, qvel)
        self.des_pos = copy.deepcopy(self.sim.data.qpos[0:7])

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
