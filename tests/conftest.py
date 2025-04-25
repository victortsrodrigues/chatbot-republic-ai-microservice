import pytest

def pytest_collection_modifyitems(config, items):
    for item in items:
        if "integration_tests" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "unit_tests" in item.nodeid:
            item.add_marker(pytest.mark.unit)