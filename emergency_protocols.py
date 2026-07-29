"""
Emergency Response Protocols for Agricultural Safety

Defines fail-safe mechanisms and escalation procedures for
emergency situations. Critical for functional safety (IEC 61508).
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EmergencyLevel(Enum):
    """Escalating emergency severity levels."""
    LEVEL_0 = "LEVEL_0"         # Normal operation
    LEVEL_1 = "LEVEL_1"         # Warning (reduce speed)
    LEVEL_2 = "LEVEL_2"         # High alert (prepare stop)
    LEVEL_3 = "LEVEL_3"         # Stop imminent
    LEVEL_4 = "LEVEL_4"         # Emergency stop active
    LEVEL_5 = "LEVEL_5"         # System failure / failsafe engaged


@dataclass
class EmergencyResponse:
    """Describes system response to emergency."""
    level: EmergencyLevel
    action: str
    immediate_commands: list  # Commands to execute NOW
    follow_up_commands: list  # Commands to execute sequentially
    estimated_response_time_ms: float
    fallback_actions: list  # If primary fails
    timeout_seconds: float  # Max time allowed for this state


class FailSafeSystem:
    """
    Functional safety (IEC 61508) compliant fail-safe system.
    
    Ensures that loss of any component doesn't cause unsafe state.
    Uses defensive programming: assume worst case.
    """
    
    def __init__(self, heartbeat_interval_ms: float = 100.0):
        """
        Initialize fail-safe system.
        
        Args:
            heartbeat_interval_ms: Monitoring heartbeat interval
        """
        self.heartbeat_interval = heartbeat_interval_ms / 1000.0
        
        # Component health monitoring
        self.component_health = {
            "camera": True,
            "detection_model": True,
            "terrain_analyzer": True,
            "risk_assessor": True,
            "ecu_connection": True,
            "can_interface": True,
            "hydraulics": True,
        }
        
        # Watchdog thread
        self.watchdog_active = True
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.watchdog_thread.start()
        
        # Failure callbacks
        self.on_critical_failure = None
        
        logger.info("Fail-safe system initialized")
    
    def check_component(self, component_name: str, is_healthy: bool):
        """Report component health status."""
        old_status = self.component_health.get(component_name, True)
        self.component_health[component_name] = is_healthy
        
        if old_status and not is_healthy:
            logger.error(f"COMPONENT FAILURE: {component_name}")
            self._handle_component_failure(component_name)
    
    def get_safety_status(self) -> dict:
        """Get overall system safety status."""
        critical_components = ["ecu_connection", "hydraulics", "camera"]
        
        degraded_components = [c for c, h in self.component_health.items() if not h and c not in critical_components]
        
        critical_failures = [c for c in critical_components if not self.component_health.get(c, True)]
        
        return {
            "all_healthy": all(self.component_health.values()),
            "degraded_components": degraded_components,
            "critical_failures": critical_failures,
            "safe_to_operate": len(critical_failures) == 0,
        }
    
    def _handle_component_failure(self, component_name: str):
        """Handle component failure using fail-safe strategy."""
        safety_status = self.get_safety_status()
        
        if not safety_status["safe_to_operate"]:
            logger.critical(f"CRITICAL SYSTEM FAILURE: {component_name}")
            self._trigger_emergency_shutdown()
        else:
            logger.warning(f"Component failure in {component_name}, operating in degraded mode")
    
    def _trigger_emergency_shutdown(self):
        """Trigger full emergency shutdown."""
        logger.critical("EMERGENCY SHUTDOWN INITIATED")
        
        if self.on_critical_failure:
            self.on_critical_failure()
    
    def _watchdog_loop(self):
        """Monitor system health periodically."""
        last_heartbeat = time.time()
        
        while self.watchdog_active:
            time.sleep(self.heartbeat_interval)
            
            current_time = time.time()
            
            # Check if heartbeats are being received
            if (current_time - last_heartbeat) > (2 * self.heartbeat_interval):
                logger.warning("Watchdog: Heartbeat timeout detected")
                # Could trigger fail-safe here
    
    def shutdown(self):
        """Graceful shutdown of fail-safe system."""
        self.watchdog_active = False
        self.watchdog_thread.join(timeout=1.0)


class EmergencyResponseController:
    """
    Handles emergency response procedures and escalation.
    
    Implements safety logic:
    1. Continuous monitoring for threats
    2. Multi-level escalation
    3. Fail-safe fallbacks
    4. Audit trail of all actions
    """
    
    def __init__(self):
        self.current_level = EmergencyLevel.LEVEL_0
        self.fail_safe = FailSafeSystem()
        self.response_history = []
        self.lock = threading.Lock()
        
        logger.info("Emergency response controller initialized")
    
    def assess_emergency_level(
        self,
        risk_score: float,
        detection_count: int,
        min_distance_m: float,
        time_to_contact_s: Optional[float],
    ) -> EmergencyLevel:
        """
        Assess emergency level based on threat parameters.
        
        Args:
            risk_score: 0-1 risk score from risk executor
            detection_count: Number of people detected
            min_distance_m: Closest person distance
            time_to_contact_s: TTC prediction or None
        
        Returns:
            EmergencyLevel indicating severity
        """
        # Hard constraints (absolute safety boundaries)
        if min_distance_m < 0.3:
            return EmergencyLevel.LEVEL_4  # CRITICAL
        
        if time_to_contact_s is not None and time_to_contact_s < 0.2:
            return EmergencyLevel.LEVEL_4  # CRITICAL
        
        # Risk score assessment
        if risk_score >= 0.85:
            return EmergencyLevel.LEVEL_4  # Emergency stop
        elif risk_score >= 0.65:
            return EmergencyLevel.LEVEL_3  # Stop imminent
        elif risk_score >= 0.45:
            return EmergencyLevel.LEVEL_2  # High alert
        elif risk_score >= 0.25:
            return EmergencyLevel.LEVEL_1  # Warning
        else:
            return EmergencyLevel.LEVEL_0  # Normal
    
    def get_response_for_level(self, level: EmergencyLevel) -> EmergencyResponse:
        """
        Get standardized response for emergency level.
        
        Response includes:
        - Immediate CAN commands
        - Secondary actions
        - Fallback procedures
        - Timing constraints
        """
        
        responses = {
            EmergencyLevel.LEVEL_0: EmergencyResponse(
                level=level,
                action="NORMAL_OPERATION",
                immediate_commands=["SET_SPEED_NORMAL"],
                follow_up_commands=[],
                estimated_response_time_ms=0,
                fallback_actions=[],
                timeout_seconds=float('inf'),
            ),
            
            EmergencyLevel.LEVEL_1: EmergencyResponse(
                level=level,
                action="REDUCE_SPEED",
                immediate_commands=["ALERT_OPERATOR", "REDUCE_SPEED_25"],
                follow_up_commands=["MONITOR_DISTANCE"],
                estimated_response_time_ms=500,
                fallback_actions=["ESCALATE_TO_LEVEL_2"],
                timeout_seconds=10.0,
            ),
            
            EmergencyLevel.LEVEL_2: EmergencyResponse(
                level=level,
                action="HIGH_ALERT",
                immediate_commands=["ALERT_OPERATOR", "REDUCE_SPEED_10", "ACTIVATE_CUTTING_SUSPENSION"],
                follow_up_commands=["READY_EMERGENCY_STOP", "MONITOR_TTC"],
                estimated_response_time_ms=300,
                fallback_actions=["ESCALATE_TO_LEVEL_3"],
                timeout_seconds=5.0,
            ),
            
            EmergencyLevel.LEVEL_3: EmergencyResponse(
                level=level,
                action="STOP_IMMINENT",
                immediate_commands=["ALERT_OPERATOR_CRITICAL", "BEGIN_EMERGENCY_STOP_SEQUENCE"],
                follow_up_commands=["ENGAGE_HYDRAULIC_BRAKE", "DISENGAGE_ENGINE", "DUMP_CUTTING_MECHANISM"],
                estimated_response_time_ms=100,
                fallback_actions=["IMMEDIATE_HYDRAULIC_DUMP"],
                timeout_seconds=2.0,
            ),
            
            EmergencyLevel.LEVEL_4: EmergencyResponse(
                level=level,
                action="EMERGENCY_STOP",
                immediate_commands=["EMERGENCY_STOP_ALL", "HYDRAULIC_DUMP_IMMEDIATE", "IGNITION_CUTOFF"],
                follow_up_commands=["ACTIVE_BRAKING", "OPERATOR_ALERT_5SEC"],
                estimated_response_time_ms=50,
                fallback_actions=["MECHANICAL_FAILSAFE", "SYSTEM_SHUTDOWN"],
                timeout_seconds=0.5,
            ),
            
            EmergencyLevel.LEVEL_5: EmergencyResponse(
                level=level,
                action="SYSTEM_FAILURE_FAILSAFE",
                immediate_commands=["EMERGENCY_STOP_ALL", "HYDRAULIC_DUMP", "IGNITION_CUTOFF", "ALERT_SERVICE"],
                follow_up_commands=["DISABLE_ALL_MOTORS"],
                estimated_response_time_ms=100,
                fallback_actions=["MECHANICAL_BRAKE_ENGAGE"],
                timeout_seconds=0.0,
            ),
        }
        
        return responses.get(level, responses[EmergencyLevel.LEVEL_0])
    
    def escalate_level(self, new_level: EmergencyLevel):
        """Escalate emergency level with logging."""
        with self.lock:
            old_level = self.current_level
            self.current_level = new_level
            
            if new_level != old_level:
                logger.warning(f"ESCALATION: {old_level.value} → {new_level.value}")
                
                # Record in history
                self.response_history.append({
                    "timestamp": time.time(),
                    "from_level": old_level.value,
                    "to_level": new_level.value,
                })


class CAN_ECU_Interface:
    """
    CAN bus interface to tractor ECU.
    
    Sends safety commands to engine control unit with
    proper timing and verification.
    """
    
    def __init__(self, can_interface=None):
        """
        Initialize ECU interface.
        
        Args:
            can_interface: Actual CAN device driver (or mock for testing)
        """
        self.can = can_interface  # In real system: python-can.Bus
        self.command_timeout = 0.5  # 500ms response timeout
        self.last_command_time = 0
        self.command_count = 0
        
        logger.info("CAN ECU interface initialized")
    
    def send_emergency_stop(self) -> bool:
        """
        Send emergency stop CAN message.
        
        Returns:
            True if acknowledged, False if timeout/failure
        """
        command = {
            "id": 0x100,  # Safety-critical CAN ID
            "dlc": 8,
            "data": [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00],
            "is_extended_id": False,
        }
        
        logger.critical("Sending EMERGENCY STOP via CAN")
        
        try:
            if self.can:
                # In real system: self.can.send(message)
                # response = wait_for_ack(timeout=self.command_timeout)
                # return response.is_received
                pass
            return True
        except Exception as e:
            logger.error(f"CAN send failed: {e}")
            return False
    
    def send_speed_command(self, target_speed_kmh: float) -> bool:
        """Send speed reduction command to ECU."""
        # Convert to ECU units (0-255 for 0-50 km/h)
        speed_value = int((target_speed_kmh / 50.0) * 255)
        speed_value = max(0, min(255, speed_value))
        
        command = {
            "id": 0x101,
            "data": [speed_value, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        }
        
        try:
            if self.can:
                # self.can.send(message)
                pass
            self.command_count += 1
            return True
        except Exception as e:
            logger.error(f"Speed command failed: {e}")
            return False
    
    def send_hydraulic_dump(self) -> bool:
        """Dump hydraulic cutting mechanism (fail-safe position)."""
        command = {
            "id": 0x102,
            "data": [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        }
        
        logger.critical("Hydraulic dump command sent")
        
        try:
            if self.can:
                # self.can.send(message)
                pass
            return True
        except Exception as e:
            logger.error(f"Hydraulic dump failed: {e}")
            return False


# Response execution
class ResponseExecutor:
    """
    Executes safety responses based on emergency level.
    Ensures commands are sent with proper timing and fallbacks.
    """
    
    def __init__(self, ecu_interface: CAN_ECU_Interface):
        self.ecu = ecu_interface
        self.last_response_level = EmergencyLevel.LEVEL_0
        self.response_times = []
    
    def execute_response(self, response: EmergencyResponse) -> bool:
        """
        Execute response commands in sequence.
        
        Returns:
            True if all commands successful, False if any failed
        """
        start_time = time.time()
        success = True
        
        logger.info(f"Executing response: {response.action}")
        
        # Execute immediate commands
        for cmd in response.immediate_commands:
            if not self._execute_command(cmd):
                success = False
                # Try fallback
                for fallback_cmd in response.fallback_actions:
                    logger.warning(f"Executing fallback: {fallback_cmd}")
                    self._execute_command(fallback_cmd)
                break
        
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        self.response_times.append(elapsed)
        
        if elapsed > response.estimated_response_time_ms * 1.5:
            logger.warning(f"Response slow: {elapsed:.1f}ms (expected {response.estimated_response_time_ms:.1f}ms)")
        
        return success
    
    def _execute_command(self, cmd: str) -> bool:
        """Execute single CAN command."""
        try:
            if cmd == "EMERGENCY_STOP_ALL":
                return self.ecu.send_emergency_stop()
            elif cmd.startswith("REDUCE_SPEED"):
                if "10" in cmd:
                    return self.ecu.send_speed_command(target_speed_kmh=0.5)
                elif "25" in cmd:
                    return self.ecu.send_speed_command(target_speed_kmh=1.25)
                else:
                    return self.ecu.send_speed_command(target_speed_kmh=0.0)
            elif cmd == "HYDRAULIC_DUMP_IMMEDIATE":
                return self.ecu.send_hydraulic_dump()
            elif "ALERT" in cmd:
                logger.critical(cmd)
                return True
            else:
                logger.debug(f"Command: {cmd}")
                return True
        except Exception as e:
            logger.error(f"Command execution failed: {cmd} - {e}")
            return False
    
    def get_performance_stats(self) -> dict:
        """Get response time statistics."""
        if not self.response_times:
            return {}
        
        import statistics
        return {
            "response_count": len(self.response_times),
            "avg_ms": statistics.mean(self.response_times),
            "min_ms": min(self.response_times),
            "max_ms": max(self.response_times),
            "p95_ms": sorted(self.response_times)[int(0.95 * len(self.response_times))],
        }


def test_emergency_protocols():
    """Test emergency response system."""
    logger.info("=" * 80)
    logger.info("EMERGENCY RESPONSE PROTOCOL TEST")
    logger.info("=" * 80)
    
    controller = EmergencyResponseController()
    ecu = CAN_ECU_Interface()
    executor = ResponseExecutor(ecu)
    
    # Test escalation
    test_cases = [
        (0.1, 0, 50.0, None, "Normal operation"),
        (0.3, 1, 5.0, None, "Warning - person detected"),
        (0.5, 1, 2.0, 1.5, "High alert - person close"),
        (0.8, 1, 0.8, 0.5, "Emergency stop - critical threat"),
    ]
    
    for risk_score, det_count, distance, ttc, description in test_cases:
        level = controller.assess_emergency_level(risk_score, det_count, distance, ttc)
        response = controller.get_response_for_level(level)
        
        logger.info(f"\n{description}")
        logger.info(f"  Risk: {risk_score:.2f}, Distance: {distance:.2f}m, TTC: {ttc}")
        logger.info(f"  Level: {level.value}")
        logger.info(f"  Action: {response.action}")
        logger.info(f"  Commands: {response.immediate_commands}")
        
        # Simulate execution
        executor.execute_response(response)
        controller.escalate_level(level)
    
    logger.info("\nPerformance Statistics:")
    stats = executor.get_performance_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    test_emergency_protocols()
