#!/usr/bin/env python3
import os
import sys
import time
import uuid
import asyncio
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import httpx
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import init_db, close_db, row_to_dict

# --- 配置 ---
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
# --- 配置结束 ---


# --- 从 app.py 移植的核心函数 ---

OIDC_BASE = "https://oidc.us-east-1.amazonaws.com"
TOKEN_URL = f"{OIDC_BASE}/token"

def _get_proxies() -> Optional[Dict[str, str]]:
    """从环境变量读取HTTP代理设置。"""
    proxy = os.getenv("HTTP_PROXY", "").strip()
    if proxy:
        return {"http": proxy, "https": proxy}
    return None

def _oidc_headers() -> Dict[str, str]:
    """构造OIDC请求所需的HTTP头。"""
    return {
        "content-type": "application/json",
        "user-agent": "aws-sdk-rust/1.3.9 os/macos lang/rust/1.87.0 exec-env/CLI md/appVersion-1.19.7",
        "x-amz-user-agent": "aws-sdk-rust/1.3.9 ua/2.1 api/ssooidc/1.88.0 os/macos lang/rust/1.87.0 exec-env/CLI m/E md/appVersion-1.19.7 app/AmazonQ-For-CLI",
        "amz-sdk-request": "attempt=1; max=3",
        "amz-sdk-invocation-id": str(uuid.uuid4()),
    }

async def refresh_single_account_token(
    db,
    account: Dict[str, Any],
    client: httpx.AsyncClient
) -> bool:
    """
    尝试为单个账户刷新accessToken。
    如果成功，更新数据库并返回 True。
    如果失败，更新失败状态并返回 False。
    """
    account_id = account["id"]
    label = account.get("label") or account_id[:8]

    if not all(k in account for k in ["clientId", "clientSecret", "refreshToken"]):
        print(f"  [!] 账号 {label} 缺少刷新所需凭证，跳过。")
        return False

    payload = {
        "grantType": "refresh_token",
        "clientId": account["clientId"],
        "clientSecret": account["clientSecret"],
        "refreshToken": account["refreshToken"],
    }

    try:
        r = await client.post(TOKEN_URL, headers=_oidc_headers(), json=payload)
        r.raise_for_status()
        data = r.json()

        new_access = data.get("accessToken")
        new_refresh = data.get("refreshToken", account.get("refreshToken"))
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

        # 刷新成功: 启用账号，重置错误计数，更新token
        await db.execute(
            """
            UPDATE accounts
            SET accessToken=?, refreshToken=?, last_refresh_time=?, last_refresh_status=?,
                updated_at=?, enabled=1, error_count=0
            WHERE id=?
            """,
            (new_access, new_refresh, now, "success", now, account_id),
        )
        print(f"  [✅] 账号 {label} 刷新成功并已重新启用。")
        return True

    except httpx.HTTPError as e:
        error_detail = str(e)
        try:
            # 尝试解析更详细的错误信息
            error_detail = e.response.json().get("error_description", str(e))
        except Exception:
            pass

        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        await db.execute(
            "UPDATE accounts SET last_refresh_time=?, last_refresh_status=?, updated_at=? WHERE id=?",
            (now, "failed", now, account_id),
        )
        print(f"  [❌] 账号 {label} 刷新失败: {error_detail}")
        return False
    except Exception as e:
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        await db.execute(
            "UPDATE accounts SET last_refresh_time=?, last_refresh_status=?, updated_at=? WHERE id=?",
            (now, "failed", now, account_id),
        )
        print(f"  [❌] 账号 {label} 发生未知错误: {e}")
        traceback.print_exc()
        return False


async def main():
    """脚本主逻辑。"""
    db = await init_db()

    # 查找目标账号
    accounts_to_retry = await db.fetchall(
        "SELECT * FROM accounts WHERE enabled = 0 AND last_refresh_status = 'failed'"
    )
    accounts_to_retry = [row_to_dict(acc) for acc in accounts_to_retry]

    if not accounts_to_retry:
        print("没有找到因刷新失败而被禁用的账号。")
        await close_db()
        return

    print(f"找到 {len(accounts_to_retry)} 个需要重试的账号...")

    success_count = 0
    failure_count = 0

    # 创建一个共享的HTTP客户端
    proxies = _get_proxies()
    mounts = None
    if proxies:
        proxy_url = proxies.get("https") or proxies.get("http")
        if proxy_url:
            mounts = {
                "https://": httpx.AsyncHTTPTransport(proxy=proxy_url),
                "http://": httpx.AsyncHTTPTransport(proxy=proxy_url),
            }

    async with httpx.AsyncClient(mounts=mounts, timeout=60.0) as client:
        for i, account in enumerate(accounts_to_retry):
            label = account.get("label") or account.get("id", "未知ID")[:8]
            print(f"\n--- ({i+1}/{len(accounts_to_retry)}) 正在处理账号: {label} ---")

            is_success = await refresh_single_account_token(db, account, client)
            if is_success:
                success_count += 1
            else:
                failure_count += 1

            # 在账号之间添加短暂延迟，避免请求过于集中
            if i < len(accounts_to_retry) - 1:
                await asyncio.sleep(1)

    await close_db()

    print("\n--- 操作完成 ---")
    print(f"成功启用: {success_count} 个账号")
    print(f"保持禁用: {failure_count} 个账号")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n操作被用户中断。")
        sys.exit(1)
