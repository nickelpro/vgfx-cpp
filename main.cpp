#include "vg.hpp"

int main() {
  vg::Window window {"Test Window", 500, 500};
  vg::Renderer renderer {window};

  window.run_continuous([&]() { renderer.draw(); });

  renderer.destroy();
  window.destroy();
}
