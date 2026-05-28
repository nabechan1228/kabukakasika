"""
入力バリデーションとセキュアなファイルパス生成ユーティリティ。

セキュリティ上重要: 銘柄コードをファイルパスやSQLクエリに使用する前に
必ずこのモジュールのバリデーション関数を通すこと。
"""
import os
import re
from fastapi import HTTPException


# ---------------------------------------------------------
# 銘柄コードバリデーション
# ---------------------------------------------------------
# 日本株の銘柄コードは4桁の数字
_STOCK_CODE_PATTERN = re.compile(r"^\d{4}$")


def validate_stock_code(code: str) -> str:
    """
    銘柄コードが4桁の数字であることを検証し、安全なコードを返す。

    Raises:
        HTTPException(400): 不正な銘柄コードの場合
    """
    if not _STOCK_CODE_PATTERN.match(code):
        raise HTTPException(
            status_code=400,
            detail="銘柄コードは4桁の数字で入力してください"
        )
    return code


# ---------------------------------------------------------
# セキュアなファイルパス生成
# ---------------------------------------------------------
# モデルとスケーラーを保存するベースディレクトリ
_MODEL_BASE_DIR = os.getenv("MODEL_DIR", os.path.dirname(os.path.abspath(__file__)))


def get_model_path(code: str) -> str:
    """
    バリデーション済みの銘柄コードからモデルファイルパスを安全に生成する。

    パストラバーサル攻撃を防ぐため、生成されたパスがベースディレクトリ内に
    収まることを検証する。

    Args:
        code: バリデーション済みの4桁銘柄コード

    Returns:
        安全な絶対ファイルパス

    Raises:
        HTTPException(400): パストラバーサルが検出された場合
    """
    filename = f"lstm_model_{code}.pth"
    full_path = os.path.normpath(os.path.join(_MODEL_BASE_DIR, filename))
    _assert_safe_path(full_path)
    return full_path


def get_scaler_path(code: str) -> str:
    """
    バリデーション済みの銘柄コードからスケーラーファイルパスを安全に生成する。

    Args:
        code: バリデーション済みの4桁銘柄コード

    Returns:
        安全な絶対ファイルパス

    Raises:
        HTTPException(400): パストラバーサルが検出された場合
    """
    filename = f"scaler_{code}.json"
    full_path = os.path.normpath(os.path.join(_MODEL_BASE_DIR, filename))
    _assert_safe_path(full_path)
    return full_path


def _assert_safe_path(path: str) -> None:
    """パスがベースディレクトリ内に収まっていることを検証する。"""
    base = os.path.normpath(_MODEL_BASE_DIR)
    if not path.startswith(base):
        raise HTTPException(
            status_code=400,
            detail="不正なファイルパスが検出されました"
        )
