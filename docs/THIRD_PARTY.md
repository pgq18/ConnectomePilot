# Third-party references

The graph preparation recipe and simplified dynamics are adapted from the
public fly.ai source by alextitonis. This application supplies its own robot
world, encoder/readout mapping, UI, tests, and residual-control interface.

Source: https://github.com/alextitonis/fly.ai

MIT License

Copyright (c) 2026 alextitonis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

MaleCNS dataset licensing is separate: CC BY 4.0. See ../data/README.md.

## Algorithmic references

- [FlyDoom](https://github.com/eganeganegan/flydoom), reference commit
  `b047fabffb62e10268de2fc1f12c765b8b019c0b`: fixed-topology continuous neural
  state, PPO over permitted connection strengths, and reward-modulated
  three-factor rules. `flylab/plasticity.py` independently adapts these ideas to
  continuous robot residual actions. The referenced tree has no standalone
  LICENSE file; this repository does not redistribute its source files.
- [fly-craftax](https://github.com/liuzihe02/fly-craftax): frozen population
  readout experiments.
- [CleanRL](https://github.com/vwxyzjn/cleanrl): recurrent/continuous PPO design
  references. The local experiment scripts implement their own rollout and
  evaluation logic.
- The trainable-dynamics experiment discusses FLYNN and FlyGM in
  [LEARNING_PROTOCOL.md](LEARNING_PROTOCOL.md); it is an independent robotics
  experiment, not a reproduction of their reported benchmark results.

Original ConnectomePilot code is covered by the root [MIT license](../LICENSE).
The fly.ai notice above applies to the adapted graph preparation and early
dynamics. Third-party data, software dependencies, fonts and media remain under
their respective licenses. Raw connectome files and pretrained model artifacts
are not bundled in the source repository.
