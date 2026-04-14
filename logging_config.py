'''
 * @Author       : MatthewZhang
 * @Date         : 2026-04-08 15:21:52
 * @Description  : 
'''
# logging_config.py
import logging

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )