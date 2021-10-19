import numpy as np

def gather_stats(agent, env):
  """ Compute average rewards over 10 episodes
  """
  score = []
  grid = []
  path = []
  for k in range(10):
      old_state = env.reset()
      cumul_r, done = 0, False
      cumul_grid, cumul_path = 0, 0
      while not done:
          a = agent.policy_action(old_state)
          old_state, r, done, info = env.step(a)
          cumul_r += r
          cumul_grid += info["grid"]
          cumul_path += info["path"]
      score.append(cumul_r)
      grid.append(cumul_grid)
      path.append(cumul_path)

  return np.mean(np.array(score)), np.std(np.array(score)), \
         np.mean(np.array(grid, np.float32)), np.std(np.array(grid, np.float32)),\
         np.mean(np.array(path)), np.std(np.array(path))
