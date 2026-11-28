import unittest

from streamlit.testing.v1 import AppTest


class NetworkBuilderTests(unittest.TestCase):
    def test_click_edit_duplicate_reorder_remove_and_restore(self):
        app = AppTest.from_string("""
import streamlit as st
from app.cnn_teaching_lab import network_builder
from app.image_processing import Layer, make_samples
if "layers" not in st.session_state:
    st.session_state.layers = [Layer(name="First"), Layer(name="Second")]
    st.session_state.image_choice = "Horizontal gradient"
network_builder(make_samples(8)["Horizontal gradient"], False, 25, False)
""", default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertNotIn("Configure block", [box.label for box in app.selectbox])
        first, second = app.session_state["layers"]
        app.button(key=f"select_{second.id}").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["selected_block"], second.id)
        app.text_input(key=f"{second.id}_name").set_value("Renamed").run()
        self.assertEqual(app.session_state["layers"][1].name, "Renamed")
        app.button(key=f"up_{second.id}").click().run()
        self.assertEqual([layer.id for layer in app.session_state["layers"]], [second.id, first.id])
        self.assertEqual(app.session_state["selected_block"], second.id)
        next(button for button in app.button if button.label == "Duplicate block").click().run()
        duplicate = app.session_state["layers"][1]
        self.assertEqual(duplicate.name, "Renamed copy")
        self.assertEqual(app.session_state["selected_block"], duplicate.id)
        next(button for button in app.button if button.label == "Remove block").click().run()
        self.assertEqual(len(app.session_state["layers"]), 2)
        self.assertEqual(app.session_state["selected_block"], first.id)
        self.assertEqual(len(app.get("iframe")), 1)
        app.toggle(key="network_show_only_output").set_value(True).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("iframe")), 0)
        next(button for button in app.button if button.label == "Start empty").click().run()
        self.assertEqual(app.session_state["layers"], [])
        self.assertIsNone(app.session_state["selected_block"])
        next(button for button in app.button if button.label == "＋ Add block").click().run()
        self.assertEqual(len(app.session_state["layers"]), 1)
        self.assertEqual(app.session_state["selected_block"], app.session_state["layers"][0].id)
        next(button for button in app.button if button.label == "Restore 10-layer example").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["layers"]), 10)
        self.assertEqual(app.session_state["selected_block"], app.session_state["layers"][0].id)


if __name__ == "__main__":
    unittest.main()
