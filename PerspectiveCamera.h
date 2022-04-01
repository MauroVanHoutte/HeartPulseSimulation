#pragma once
#include "glm.hpp"

class PerspectiveCamera
{
public:
	PerspectiveCamera(const glm::fvec3& position, const glm::fvec3 forwardVector, float aspectRatio);
	PerspectiveCamera(const PerspectiveCamera& other) = delete;
	PerspectiveCamera& operator=(const PerspectiveCamera& other) = delete;
	PerspectiveCamera(PerspectiveCamera&& other) = delete;
	PerspectiveCamera& operator=(PerspectiveCamera&& other) = delete;
	
	glm::fvec3 GetPosition() const;
	glm::fvec3 GetForwardVector() const;
	glm::fvec3 GetRightVector() const;
	glm::fvec3 GetUpVector() const;
	glm::mat4 GetLookAt() const;
	glm::mat4 GetViewMatrix() const;
	glm::mat4 GetProjectionMatrix() const;
	float GetAspectRatio() const;
	float GetDistance() const;
	float GetFieldOfView() const;
	float GetMovementSpeed() const;
	float GetRotationSpeed() const;
	float GetFar() const;
	float GetNear() const;

	void SetFieldOfView(float degrees);
	void SetDistance(float distance);
	void SetMovementSpeed(float speed);
	void SetRotationSpeed(float speed);
	void SetPosition(const glm::fvec3& position);
	void SetFar(float farPlane);
	void SetNear(float nearPlane);

	void Translate(const glm::fvec3& translation);
	void Rotate(float angle, const glm::fvec3& axis, bool isDegrees = true);
	void RotateYaw(float angle, bool isDegrees = true);
	void RotatePitch(float angle, bool isDegrees = true);

private:
	glm::mat4 m_LookAt;
	glm::mat4 m_ProjectionMatrix;
	float m_AspectRatio;
	float m_Fov;
	float m_Distance;
	float m_Angle;
	float m_MovementSpeed;
	float m_RotationSpeed;
	float m_Far;
	float m_Near;

	void SetForwardVector(const glm::fvec3& forwardVector);
	void CalculateProjectionMatrix();
	void CalculateVectors();
};

